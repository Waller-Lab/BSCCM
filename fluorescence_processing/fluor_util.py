import numpy as np
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

    def __init__(self, dataframe, initial_mask, channel_names, 
                 img_fig=None, read_image_fn=None, num_cols=2, export_dir=None):
        """
        
        figure: the main figure
        dataframe: pandas dataframe containing the data
        mask: mask for indices within the full dataframe
        channel names: list of channel names
        img_fig: existing figure on which stuff is done
        read_image_fn: for making montages, function that takes an index as an arg and returns and image
        num_cols: number of columns to lay out successive facs plots
        """
        self.fig = plt.figure()
        self.img_fig = img_fig
        
        self.num_cols = num_cols
        self.channel_names = channel_names
        self.dataframe = dataframe
        self.read_image_fn = read_image_fn
        self.channel_names = channel_names
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
                    #make a new terminal subplot
                    self.add_subplot(current_subplot) 
                else:
                    # intermediate selection updated
                    self.selections_updated()

        gate_button = widgets.Button(description='Gate selection')                
        gate_button.on_click(do_gating) 
        button_list.append(gate_button)

        
        # Button for showing image montage
        if read_image_fn is not None:
            def show_montage(Button):
                self.show_selection_montage()
            montage_button = widgets.Button(description='Show images')                
            montage_button.on_click(show_montage)
            button_list.append(montage_button)
                              
        
        # Button for exporting plot
        if export_dir is not None:
            box = widgets.Text(
                value='Export_name.pdf',
                placeholder='',
                description=''
            )

            def export_fig(button):
                self.fig.savefig(export_dir + box.value, transparent=True)
                print('Saved to {}{}'.sformat(export_dir, box.value))

            export_button = widgets.Button(description='Export figure')                
            export_button.on_click(export_fig) 
            button_list.append(box)
            button_list.append(export_button)
        
        
        display(widgets.HBox(button_list))
        
        ### Controls for marking populations in the dataframe
        button_list_2 = []
        selection_box = widgets.Text(
            value='name',
            placeholder='',
            description=''
        )

        def select_population(button):
            print('Saving selection...')
            current_subplot = self.subplots[self.plot_index_button.value]
            if hasattr(current_subplot, 'path'): 
                selected_global_indices = current_subplot.finalize_selection()
                name = 'selection_' + selection_box.value
                self.dataframe.loc[selected_global_indices, name] = True
            print('Complete...')

        save_selection_button = widgets.Button(description='Save selection')                
        save_selection_button.on_click(select_population) 
        button_list_2.append(selection_box)
        button_list_2.append(save_selection_button)
        
        display(widgets.HBox(button_list_2))
        
        
        self.selected_index = None
        self.path = None
        self.canvas = self.fig.canvas  
        self.change_data(initial_mask)
        self.add_subplot(None)

    def change_data(self, mask):
        #Change the underlying data
#         self.subplots = []
#         self.fig.clf()
        
        #TODO: re add subplots with same gates but different masked data
        
        
        self.all_indices = self.dataframe.loc[np.flatnonzero(mask)].global_index.to_numpy()
        self.selections_updated()
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
            
    def selections_updated(self):
        """
        One of the subplots has had a gate change. Move through all of them to update plots
        """
        for sp in self.subplots:
            sp.update_plot()
            
    def set_active_subplot(self, subplot):
        print('setting active subplot ' + str(subplot))
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
        if parent is None: #0th plot
            self.sub_indices = main.all_indices
        else: # >= 1st plot
            self.sub_indices = parent.get_selected_indices()
        self.ch_x = ch_x
        self.ch_y = ch_y

        num_subplots = len(self.main.fig.axes)
        for i in range(num_subplots):
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
    
    def update_plot(self):
        #updat the selected points
        if self.parent is None: #0th plot
            self.sub_indices = self.main.all_indices
        else: # >= 1st plot
            self.sub_indices = self.parent.get_selected_indices()
        
        self.facs_ax.clear()
        # get data
        x_data = self.main.dataframe.loc[self.sub_indices, self.ch_x]
        y_data = self.main.dataframe.loc[self.sub_indices, self.ch_y]

        xy = np.vstack([x_data, y_data])
        #Compute point density
        kde_source_points = 1000 #trades offf accuracy and speed
        kde_indices = np.random.choice(xy.shape[1], kde_source_points)

        self.density = gaussian_kde(xy[:, kde_indices])(xy)
                
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
                                       lineprops={'color':'g'})

        #Do scatter plot
        self.collection = self.facs_ax.scatter(x_data, y_data, 
                                               c=self.density, s=20, cmap='inferno')
        self.facs_ax.ticklabel_format(scilimits=[-2, 2])

        self.facs_ax.set_xlabel(self.ch_x)
        self.facs_ax.set_ylabel(self.ch_y)
        
        all_x_data = self.main.dataframe.loc[self.main.all_indices, self.ch_x]
        all_y_data = self.main.dataframe.loc[self.main.all_indices, self.ch_y]
        self.facs_ax.set_xlim([0.8 * np.min(all_x_data), 1.1 * np.max(all_x_data)])
        self.facs_ax.set_ylim([0.8 * np.min(all_y_data), 1.1 * np.max(all_y_data)])
        
#         self.facs_ax.set_ylim([6.5, 17])
#         self.facs_ax.set_xlim([6.5, 17])
        
    def apply_color(self):
        color_norm = colors.Normalize(vmin=np.min(self.density), vmax=np.max(self.density))
        m = cm.ScalarMappable(norm=color_norm, cmap=cm.inferno)
        rgba_base = m.to_rgba(self.density)
        
        color_norm = colors.Normalize(vmin=np.min(self.density), vmax=np.max(self.density))
        m = cm.ScalarMappable(norm=color_norm, cmap=cm.winter)
        rgba_selection = m.to_rgba(self.density)
        rgba_base[self.selection] = rgba_selection[self.selection]
    
        self.collection.set_color(rgba_base)
     
    def finalize_selection(self):
        #Create a patch marking the indices selected, and return them for making a new plot
        print('finalizing selection')
   
        self.patch_ch_x, self.patch_ch_y = self.ch_x, self.ch_y
        self.selection_patch = patches.PathPatch(self.path, facecolor='none', edgecolor='cyan',lw=2)
        self.facs_ax.add_patch(self.selection_patch)
        self.main.canvas.draw_idle()
        self.main.selections_updated()
        
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
        
        