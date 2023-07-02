import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import display
from matplotlib.widgets import LassoSelector, PolygonSelector
from matplotlib.path import Path as mplPath
import matplotlib.cm as cm
import matplotlib.patches as patches
from scipy import stats
from scipy.spatial import distance
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.neighbors import KernelDensity
from scipy import interpolate
from pathlib import Path
from matplotlib import colors
import matplotlib.gridspec as gridspec
import os


class ScatterSelectorGating:
    """
    Master class that controls a bunch of subplots
    """

    def __init__(self, bsccm, channel_names,
                 read_image_fn=None, num_cols=2, export_dir=None):
        """
        
        figure: the main figure
        bsccm: BSCCM object containg data
        channel names: list of channel names
        read_image_fn: for making montages, function that takes an index as an arg and returns and image
        num_cols: number of columns to lay out successive facs plots
        """
        dataframe = bsccm.surface_marker_dataframe
        
        dimensions = {}
        dimensions['antibodies'] = bsccm.index_dataframe['antibodies'].unique().tolist()
        dimensions['batch'] = batches = bsccm.index_dataframe['batch'].unique().tolist()
        dimensions['slide_replicate']  = bsccm.index_dataframe['slide_replicate'].unique().tolist()

        self.log_plot = False
        # Setup top controls for selecting antobody/batch/slide
        selector = None
        def on_value_change(change):
            masks = [bsccm.index_dataframe[name] == widget_dict[name].value for name in widget_dict.keys()]
            total_mask = np.logical_and(np.logical_and(masks[0], masks[1]), masks[2])
            self.set_selection(None)
            self.change_data(total_mask)

        widget_dict = {}
        for name in dimensions.keys():
            widget_dict[name] = widgets.Dropdown(
                options=dimensions[name], value=dimensions[name][0],
                description=name + ':', disabled=False)
            widget_dict[name].observe(on_value_change, names='value')


        display(widgets.HBox(list(widget_dict.values())))

        masks = [bsccm.index_dataframe[name] == widget_dict[name].value for name in widget_dict.keys()]
        initial_mask = np.logical_and(np.logical_and(masks[0], masks[1]), masks[2])
        
        self.fig = plt.figure()
        
        self.num_cols = num_cols
        self.channel_names = channel_names
        self.dataframe = dataframe
        self.read_image_fn = read_image_fn
        self.subplots = []

        #Create buttons for changing plot and channel
        self.plot_index_button = widgets.ToggleButtons(options=np.arange(0), description='Plot_index')
        self.ch_x_button = widgets.ToggleButtons(options=channel_names, description='X axis')
        self.ch_y_button = widgets.ToggleButtons(options=channel_names, description='Y axis')
        #initialize
        self.ch_x_button.value = channel_names[0]
        self.ch_y_button.value = channel_names[1]

        def set_channel(plot_index, ch_x, ch_y):
            if plot_index is None:
                return
            self.subplots[plot_index].set_channels(ch_x, ch_y)
            self.update_limit_sliders(self.subplots[plot_index])
        
        #Correctly update selected channels when changing plots
        def update_display_channels(*args):
            new_plot_index = self.plot_index_button.value
            new_ch_x = self.subplots[new_plot_index].ch_x
            new_ch_y = self.subplots[new_plot_index].ch_y
            self.ch_x_button.value = new_ch_x
            self.ch_y_button.value = new_ch_y
            
        self.plot_index_button.observe(update_display_channels, 'value')
                

        _ = interact(set_channel, plot_index=self.plot_index_button,
                     ch_x=self.ch_x_button, ch_y=self.ch_y_button)
        
        #### Add additional controls
        button_list = []
        def do_gating(button):
            current_subplot = self.subplots[self.plot_index_button.value]
            if hasattr(current_subplot, 'path'): 
                current_subplot.finalize_selection()
                if self.plot_index_button.value == len(self.subplots) - 1:
#                     new_row = len(self.subplots) % num_cols == 0
                    #make a new terminal subplot
                    self.add_subplot(current_subplot) 
                    #clear selection
                    self.set_selection(None)
#                     if new_row:
#                         new_num_rows = np.ceil(len(self.subplots) / num_cols)
#                         size = self.fig.get_size_inches()
#                         new_size = [size[0], size[1] * new_num_rows / (new_num_rows - 1)]
#                         print(size, new_size)
#                         self.fig.set_size_inches(new_size)
                else:
                    # intermediate selection updated
                    self.update_all_plots()

        gate_button = widgets.Button(description='Gate selection')                
        gate_button.on_click(do_gating) 
        button_list.append(gate_button)

        clear_button = widgets.Button(description='Clear selection')                
        clear_button.on_click(lambda widget: self.set_selection(None)) 
        button_list.append(clear_button)
        
        # Button for showing image montage
        if read_image_fn is not None:
            def show_montage(Button):
                self.show_selection_montage()
            montage_button = widgets.Button(description='Show images')                
            montage_button.on_click(show_montage)
            button_list.append(montage_button)
                     
                
        display(widgets.HBox(button_list))
        
        ### Controls for marking populations in the dataframe
        button_list_2 = []
        new_selection_name = widgets.Text(
            value='name',
            placeholder='',
            description=''
        )

        def save_selected_population(button):
            current_subplot = self.subplots[self.plot_index_button.value]
            if hasattr(current_subplot, 'path'): 
                name = 'selection_' + new_selection_name.value
                self.dataframe = self.dataframe.assign(**{name: False})
                self.dataframe.loc[self.selected_indices, name] = True
                self.show_population_dropdown.options = self.show_population_dropdown.options + (new_selection_name.value,)
                
                for subplot in self.subplots:
                    subplot.apply_color()
                
                dataframe_saving_fullpath = bsccm.data_root + 'BSCCM_surface_markers.csv'
                self.dataframe.to_csv(dataframe_saving_fullpath, index=False)
                

        save_selection_button = widgets.Button(description='Save selection')                
        save_selection_button.on_click(save_selected_population) 
        
        
        names = [col.split('selection_')[-1] for col in self.dataframe.columns if 'selection_' in col]
        self.show_population_dropdown = widgets.Dropdown(
                options=names,
                value=names[0] if len(names) > 0 else None,
                description='Show:',
                disabled=False,
            )
        def dropdown_callback(widget):
            for sp in self.subplots:
                sp.apply_color()
                
        self.show_population_dropdown.observe(dropdown_callback, names='value')

     
        button_list_2.append(new_selection_name)
        button_list_2.append(save_selection_button)
        button_list_2.append(self.show_population_dropdown)
        
        display(widgets.HBox(button_list_2))

        
        button_list_3 = []

        self.manual_axes = False
        manual_axes_button = widgets.ToggleButton(
            value=False,
            description='Manual axes',
        )
        def manual_axes_fn(button):
            self.manual_axes = manual_axes_button.value
            self.x_lim_slider.disabled = not self.manual_axes
            self.y_lim_slider.disabled = not self.manual_axes
            
            if not self.manual_axes:
                self.subplots[self.plot_index_button.value].autoscale_xy_limits()
                
        manual_axes_button.observe(manual_axes_fn, names='value')
        
        button_list_3.append(manual_axes_button)        
        
        all_data = self.dataframe[self.channel_names].to_numpy()
        data_min, data_max = np.min(all_data), np.max(all_data)
        slider_min = data_min - 0.05 * (data_max - data_min)
        slider_max = data_max + 0.05 * (data_max - data_min)
    
        self.x_lim_slider = widgets.FloatRangeSlider(
            value=[0, 1],
            disabled=True,
            min=slider_min,
            max=slider_max,
            step=1,
            description='x range:',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )
        
        self.y_lim_slider = widgets.FloatRangeSlider(
            value=[0, 1],
            disabled=True,
            min=slider_min,
            max=slider_max,
            step=1,
            description='y range:',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )
        
        def set_x_limits_fn(widget):
            if not self.manual_axes:
                return
            self.subplots[self.plot_index_button.value].facs_ax.set_xlim(
                  (self.x_lim_slider.value[0], self.x_lim_slider.value[1])) 

        def set_y_limits_fn(widget):
            if not self.manual_axes:
                return
            self.subplots[self.plot_index_button.value].facs_ax.set_ylim(
                  (self.y_lim_slider.value[0], self.y_lim_slider.value[1])) 

        self.x_lim_slider.observe(set_x_limits_fn, names='value')
        self.y_lim_slider.observe(set_y_limits_fn, names='value')
        
        button_list_3.append(self.x_lim_slider)
        button_list_3.append(self.y_lim_slider)
        
        
        display(widgets.HBox(button_list_3))
        
        button_list_4 = []
        self.density_scaling_slider = widgets.FloatLogSlider(
                value=1,
                base=10,
                min=-2,
                max=0,
                step=0.01,
                description='Density scale:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.2f',
            )
      
        def denisty_scale_fn(button):
            for sp in self.subplots:
                sp.apply_color()
        self.density_scaling_slider.observe(denisty_scale_fn, names='value')
        button_list_4.append(self.density_scaling_slider)
        
        self.log_plot = False
        log_plot_button = widgets.ToggleButton(
            value=self.log_plot,
            description='Log plot',
        )
        def log_plot_fn(button):
            self.log_plot = log_plot_button.value
            
            #Change range of limit sliders
            if self.log_plot:
                log_data_min = self.log_min_slider.value
                data_min, data_max = np.log(np.min(all_data + log_data_min)), np.log(np.max(all_data) + log_data_min)
                slider_min = data_min - 0.05 * (data_max - data_min)
                slider_max = data_max + 0.05 * (data_max - data_min)
                self.x_lim_slider.step = 0.01
                self.y_lim_slider.step = 0.01
                self.x_lim_slider.readout_format='.2f'
                self.y_lim_slider.readout_format='.2f'
            else:
                data_min, data_max = np.min(all_data), np.max(all_data)
                slider_min = data_min - 0.05 * (data_max - data_min)
                slider_max = data_max + 0.05 * (data_max - data_min)
                self.x_lim_slider.step = 1
                self.y_lim_slider.step = 1
                self.x_lim_slider.readout_format='d'
                self.y_lim_slider.readout_format='d'
            self.x_lim_slider.max = slider_max
            self.y_lim_slider.max = slider_max
            self.x_lim_slider.min = slider_min
            self.y_lim_slider.min = slider_min
            
            self.update_all_plots()
            
            if not self.manual_axes:
                self.subplots[self.plot_index_button.value].autoscale_xy_limits()
                
        log_plot_button.observe(log_plot_fn, names='value')
        button_list_4.append(log_plot_button) 
        
        self.log_min_slider = widgets.FloatLogSlider(
                value=-6,
                base=10,
                min=-6,
                max=2,
                step=0.1,
                description='Log min:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.3f',
            )
      
       
        self.log_min_slider.observe(log_plot_fn, names='value')
        button_list_4.append(self.log_min_slider)
        
    
        display(widgets.HBox(button_list_4))
        
        # Button for exporting plot
        if export_dir is not None:
            export_text_box = widgets.Text(
                value='Export_name.pdf',
                placeholder='',
                description=''
            )

            def export_fig_fig(button):
                self.fig.savefig(export_dir + export_text_box.value, transparent=True)
                print('Saved to {}{}'.format(export_dir, export_text_box.value))

            export_button = widgets.Button(description='Export figure')                
            export_button.on_click(export_fig_fig) 
            
            display(widgets.HBox([export_text_box, export_button]))

        
        self.selected_index = None
        self.path = None
        self.canvas = self.fig.canvas  
        self.change_data(initial_mask)
        self.add_subplot(None)
        
        
        #Creat a figure for visualizing images and add a button for exporting
        self.img_fig = plt.figure(figsize=(4,4))
        export_text_box_img = widgets.Text(
                        value='Export_name.pdf',
                        placeholder='',
                        description=''
                    )

        def export_fig(button):
            self.img_fig.savefig(export_dir + export_text_box_img.value, transparent=True)
            print('Saved to {}{}'.format(export_dir, export_text_box_img.value))

        export_button = widgets.Button(description='Export figure')                
        export_button.on_click(export_fig) 

        display(widgets.HBox([export_text_box_img, export_button]))

        #intialize
        on_value_change(None)
        
    def update_limit_sliders(self, caller, xlim=None, ylim=None):
#         print(xlim, ylim, self.x_lim_slider.min, self.y_lim_slider.min)
        if caller not in self.subplots:
            return #intializing
        if xlim is None:
            xlim = caller.facs_ax.get_xlim()
            ylim = caller.facs_ax.get_ylim()
        if self.subplots.index(caller) == self.plot_index_button.value:
            self.x_lim_slider.value = xlim            
            self.y_lim_slider.value = ylim
            
    def change_data(self, mask):
        #Change the underlying data
        self.all_indices = self.dataframe.loc[np.flatnonzero(mask)].index.to_numpy()

        self.update_all_plots()
#         self.add_subplot(None)
        
    def add_subplot(self, parent_subplot):
        if self.plot_index_button.value != len(self.subplots) - 1 and len(self.subplots) > 0:
            print('Must have last plot selected to select new')
            return

        self.subplots.append(ScatterSelectorSubplot(self, parent_subplot, len(self.subplots), 
                             self.ch_x_button.value, self.ch_y_button.value))
#                              self.subplots[-1] if len(self.subplots) > 0 else None ))
        
        #update plot_index button
        self.plot_index_button.options = range(len(self.subplots))
        self.plot_index_button.value = len(self.subplots) - 1
    
    def set_selection(self, indices):
        #These indices have been selected in a plot, propagate to all
        self.selected_indices = indices
        for sp in self.subplots:
            sp.set_selection(indices)
           
    def show_selection_montage(self):
        dim = (6, 6)
        gs = gridspec.GridSpec(*dim)
        gs.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
        size = dim[0] * dim[1]
        if len(self.selected_indices) >= size: 
            indices = np.random.choice(self.selected_indices, size=size, replace=False)
        else:
            indices = self.selected_indices
        images = np.stack([self.read_image_fn(i) for i in indices], axis=0)
        for index, image in enumerate(images):
            ax = self.img_fig.add_subplot(gs[index])
            ax.imshow(image, cmap='inferno')
            ax.set_axis_off()
            
    def update_all_plots(self):
        """
        One of the subplots has had a gate change. Move through all of them to update plots
        """
        for sp in self.subplots:
            sp.update_plot()
            
    def set_active_subplot(self, subplot):

        index = self.subplots.index(subplot)
        new_ch_x = self.subplots[index].ch_x
        new_ch_y = self.subplots[index].ch_y
        self.plot_index_button.value = index
        self.ch_x_button.value = new_ch_x
        self.ch_y_button.value = new_ch_y
    

class ScatterSelectorSubplot:
    """
    A single subplot within a larger figure
    """

    def __init__(self, main, parent, plot_index, ch_x, ch_y):   
        self.main = main
        self.parent = parent
        self.collection = None
        if parent is None: #0th plot
            self.sub_indices = main.all_indices
        else: # >= 1st plot
            self.sub_indices = parent.get_selected_indices()
        self.ch_x = ch_x
        self.ch_y = ch_y

        num_subplots = len(self.main.fig.axes)
        for i in range(num_subplots):
            print(self.main.fig.axes[i])
            self.main.fig.axes[i].change_geometry(num_subplots // self.main.num_cols + 1, self.main.num_cols, i + 1)
        if num_subplots == 0:
            self.facs_ax = self.main.fig.add_subplot(1,self.main.num_cols,1)
        else:
            self.facs_ax = self.main.fig.add_subplot(num_subplots // self.main.num_cols + 1, 
                                                  self.main.num_cols, num_subplots + 1)
        self.facs_ax.set_title('Plot {}'.format(plot_index))

        self.update_plot()

    def get_selected_indices(self):
        """
        return the globabl indices selected in the parents gate 
        """ 
        xys = self.collection.get_offsets()
        selection = np.nonzero(self.path.contains_points(xys))[0] 
        return self.sub_indices[selection]
        
    def set_channels(self, ch_x, ch_y):

        if ch_x == self.ch_x and ch_y == self.ch_y:
            return
        self.ch_x = ch_x
        self.ch_y = ch_y
        self.update_plot()
        self.apply_color()
        self.main.canvas.draw()
        
    def autoscale_xy_limits(self):
        all_x_data = self.main.dataframe.loc[self.main.all_indices, self.ch_x]
        all_y_data = self.main.dataframe.loc[self.main.all_indices, self.ch_y]

        if self.main.log_plot:
            min_x = np.log(np.min(all_x_data) + self.main.log_min_slider.value)
            min_y = np.log(np.min(all_y_data) + self.main.log_min_slider.value)
            max_x = np.log(np.max(all_x_data) + self.main.log_min_slider.value)
            max_y = np.log(np.max(all_y_data) + self.main.log_min_slider.value)
            range_x = max_x - min_x
            range_y = max_y - min_y
        else: 
            min_x = np.min(all_x_data)
            min_y = np.min(all_y_data)
            max_x = np.max(all_x_data)
            max_y = np.max(all_y_data)
            range_x = max_x - min_x
            range_y = max_y - min_y 
        self.facs_ax.set_xlim([min_x - 0.05 * range_x, max_x + 0.05 * range_x])
        self.facs_ax.set_ylim([min_y - 0.05 * range_y, max_y + 0.05 * range_y])
        self.main.update_limit_sliders(self, self.facs_ax.get_xlim(), self.facs_ax.get_ylim())
    
    def update_plot(self):
        #updat the selected points
        if self.parent is None: #0th plot
            self.sub_indices = self.main.all_indices
        else: # >= 1st plot
            self.sub_indices = self.parent.get_selected_indices()
        
        #Do scatter plot
        if self.collection is not None and self.main.manual_axes:
            limits = ((self.main.x_lim_slider.value[0], self.main.x_lim_slider.value[1]),
                      (self.main.y_lim_slider.value[0], self.main.y_lim_slider.value[1]))
        else: 
            limits = None
        
        self.facs_ax.clear()
        # get data
        x_data = self.main.dataframe.loc[self.sub_indices, self.ch_x]
        y_data = self.main.dataframe.loc[self.sub_indices, self.ch_y]
        
        if self.main.log_plot:
            x_data = np.log(self.main.log_min_slider.value + x_data)
            y_data = np.log(self.main.log_min_slider.value + y_data)
        xy = np.vstack([x_data, y_data])
        
        #Compute point density
        kde_source_points = 1000 #trades offf accuracy and speed
        if xy.shape[1] != 0:
            kde_indices = np.random.choice(xy.shape[1], kde_source_points)
            self.density = gaussian_kde(xy[:, kde_indices])(xy)
        else: 
            self.density= None
                
        if hasattr(self, 'selection_patch') and self.selection_patch is not None:
            self.selection_patch.remove()
            self.selection_patch = None
        if hasattr(self, 'selection_patch') and \
                self.patch_ch_x == self.ch_x and \
                self.patch_ch_y == self.ch_y:
            self.selection_patch = patches.PathPatch(self.path, 
                                       facecolor='none', edgecolor='cyan',lw=2)
            self.facs_ax.add_patch(self.selection_patch)
            
        #Create new selector
        self.selector = PolygonSelector(self.facs_ax, onselect=self.onselect,
                                       )
                                        # lineprops={'color':'g'})


        self.collection = self.facs_ax.scatter(x_data, y_data, 
                                               c=self.density, s=20, cmap='inferno', rasterized=True)
             
        self.facs_ax.ticklabel_format(scilimits=[-2, 2])

        self.facs_ax.set_xlabel(self.ch_x)
        self.facs_ax.set_ylabel(self.ch_y)
        
        if limits is not None:
            self.facs_ax.set_xlim(limits[0])
            self.facs_ax.set_ylim(limits[1])
        else:
            self.autoscale_xy_limits()
        self.main.update_limit_sliders(self, self.facs_ax.get_xlim(), self.facs_ax.get_ylim())
        
#         self.facs_ax.set_ylim([6.5, 17])
#         self.facs_ax.set_xlim([6.5, 17])
        self.apply_color()

        
    def apply_color(self):
        ## apply density mapping coloring
        min_val = np.min(self.density)
        max_val = np.max(self.density)
        max_val = (max_val - min_val) * self.main.density_scaling_slider.value

        color_norm = colors.Normalize(vmin=min_val, vmax=max_val)
        m = cm.ScalarMappable(norm=color_norm, cmap=cm.inferno)
        rgba_base = m.to_rgba(self.density)
        
    
#         min_val = np.min(self.density)
#         max_val = np.max(self.density)
#         exp = self.main.density_scaling_slider.value
#         color_norm = colors.Normalize(vmin=min_val ** exp, vmax=max_val ** exp)
#         m = cm.ScalarMappable(norm=color_norm, cmap=cm.inferno)
#         rgba_base = m.to_rgba(self.density ** exp)
        
        
        
        #special coloring for other selections
        for col_name in self.main.dataframe.columns:
            if 'selection_' in col_name:
                name = col_name.split('selection_')[-1]
                
                if self.main.show_population_dropdown.value == name:
                    mask = self.main.dataframe.loc[self.sub_indices, col_name].to_numpy()
                    rgba = np.array([0, 1, 0, 1.])
                    rgba_base[mask] = rgba
    
        #special coloring for current selection
        if hasattr(self, 'selection') and self.selection is not None:
            color_norm = colors.Normalize(vmin=np.min(self.density), vmax=np.max(self.density))
            m = cm.ScalarMappable(norm=color_norm, cmap=cm.winter)
            rgba_selection = m.to_rgba(self.density)
            rgba_base[self.selection] = rgba_selection[self.selection]
    
        self.collection.set_color(rgba_base)
        self.main.canvas.draw()
     
    def finalize_selection(self):
        #Create a patch marking the indices selected, and return them for making a new plot
   
        self.patch_ch_x, self.patch_ch_y = self.ch_x, self.ch_y
        self.selection_patch = patches.PathPatch(self.path, facecolor='none', edgecolor='cyan',lw=2)
        self.facs_ax.add_patch(self.selection_patch)
#         self.main.canvas.draw_idle()
        self.main.update_all_plots()
        
        
    def onselect(self, verts):
        #Re add first point so path closes properly
        verts = np.array(verts)
        self.path = mplPath(np.concatenate([verts, verts[0][None,:]], axis=0)) 
        
        xys = self.collection.get_offsets()
        #indices of points in path
        self.selection = np.flatnonzero(self.path.contains_points(xys))
        self.main.set_selection(self.sub_indices[self.selection])
        
        self.apply_color()
        self.main.set_active_subplot(self)
        
    def set_selection(self, indices):
        #Called by main to sync highlighted indices between plots
        self.selection = np.flatnonzero(np.isin(self.sub_indices, indices))
        self.apply_color()
        
        