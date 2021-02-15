import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
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

    def __init__(self, all_indices, data, channel_names, bsbcm, img_fig):
        self.fig = plt.figure()
        self.img_fig = img_fig
        self.data = data
        self.bsbcm = bsbcm
        self.all_indices = all_indices
        self.channel_names = channel_names
        self.subplots = []
        #Create buttons for changing plot and channel
        self.plot_index_button = widgets.ToggleButtons(
            options=np.arange(0), description='Plot_index')
        self.ch_x_button = widgets.ToggleButtons(options=channel_names, description='X axis')
        self.ch_y_button = widgets.ToggleButtons(options=channel_names, description='Y axis')
        #initialize
        self.ch_x_button.value = channel_names[0]
        self.ch_x_button.value = channel_names[1]
        
        def set_channel(plot_index, ch_x, ch_y):
            if plot_index is None:
                return
            self.subplots[plot_index].set_channel_indices(
                        channel_names.index(ch_x), channel_names.index(ch_y))
        
        #Correctly update selected channels when changing plots
        def update_display_channels(*args):
            new_plot_index = self.plot_index_button.value
            self.ch_x_button.value = channel_names[self.subplots[new_plot_index].x_index]
            self.ch_y_button.value = channel_names[self.subplots[new_plot_index].y_index]
        self.plot_index_button.observe(update_display_channels, 'value')
            
        _ = interact(set_channel, plot_index=self.plot_index_button,
                     ch_x=self.ch_x_button, ch_y=self.ch_y_button)


        self.all_indices = all_indices
        self.selected_index = None
        self.path = None
        self.canvas = self.fig.canvas

        self.add_subplot(all_indices)
        
        # Add key and mouse listeners
        def key_press(event):
            if event.key == "i":
                self.show_selection_montage()
            elif event.key == "enter":
                sp = self.subplots[self.plot_index_button.value]
                if hasattr(sp, 'path'): 
                    selected_global_indices = sp.finalize_selection()

                    self.add_subplot(selected_global_indices)     
        self.canvas.mpl_connect("key_press_event", key_press)   
        

    def add_subplot(self, indices):
        if self.plot_index_button.value != len(self.subplots) - 1 and len(self.subplots) > 0:
            print('Must have last plot selected to select new')
            return
        self.subplots.append(ScatterSelectorSubplot(self, indices, len(self.subplots), 
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
        indices = np.random.choice(self.selected_indices, size=dim[0] * dim[1], replace=False)
        images = np.stack([self.bsbcm.read_image(index=i, contrast_type='dpc')
                  for i in indices], axis=0)
        for index, image in enumerate(images):
            ax = self.img_fig.add_subplot(gs[index])
            ax.imshow(image, cmap='inferno')
            ax.set_axis_off()
    

class ScatterSelectorSubplot:
    """
    """

    def __init__(self, master, sub_indices, plot_index, ch_x, ch_y):        
        self.master = master
        self.sub_indices = sub_indices
        self.x_index = self.master.channel_names.index(ch_x)
        self.y_index = self.master.channel_names.index(ch_y)        

        num_subplots = len(self.master.fig.axes)
        for i in range(num_subplots):
            self.master.fig.axes[i].change_geometry(num_subplots // 2 + 1, 2, i + 1)
        if num_subplots == 0:
            self.facs_ax = self.master.fig.add_subplot(1,2,1)
        else:
            self.facs_ax = self.master.fig.add_subplot(num_subplots // 2 + 1, 
                                                  2, num_subplots + 1)
        self.facs_ax.set_title('Plot {}'.format(plot_index))

        self.update_plot()

    def set_channel_indices(self, x_i, y_i):
        if x_i == self.x_index and y_i == self.y_index:
            return
        self.x_index = x_i
        self.y_index = y_i
        self.update_plot()
    
    def update_plot(self):
        self.facs_ax.clear()
        # get data
        x_data = self.master.data[self.sub_indices, self.x_index]
        y_data = self.master.data[self.sub_indices, self.y_index]

        xy = np.vstack([x_data, y_data])
        #Compute point density
        self.density = gaussian_kde(xy)(xy)
                
        if hasattr(self, 'selection_patch') and self.selection_patch is not None:
            self.selection_patch.remove()
            self.selection_patch = None
        if hasattr(self, 'selection_patch') and \
                self.patch_ch_x_index == self.x_index and \
                self.patch_ch_y_index == self.y_index:
            self.selection_patch = patches.PathPatch(self.path, 
                                       facecolor='none', edgecolor='cyan',lw=2)
            self.facs_ax.add_patch(self.selection_patch)
            
        #Create new selector
        self.selector = PolygonSelector(self.facs_ax, onselect=self.onselect,
                                       lineprops={'color':'g'})

        #Do scatter plot
        self.collection = self.facs_ax.scatter(x_data, y_data, 
                                               c=self.density, s=20, cmap='inferno')

        self.facs_ax.set_xlabel(self.master.channel_names[self.x_index])
        self.facs_ax.set_ylabel(self.master.channel_names[self.y_index])
        self.facs_ax.set_ylim([6.5, 17])
        self.facs_ax.set_xlim([6.5, 17])
        
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
        self.patch_ch_x_index, self.patch_ch_y_index = self.x_index, self.y_index
        self.selection_patch = patches.PathPatch(self.path, facecolor='none', edgecolor='cyan',lw=2)
        self.facs_ax.add_patch(self.selection_patch)
        self.master.canvas.draw_idle()
        #now find which indices are contained in
        xys = self.collection.get_offsets()
        selection = np.nonzero(self.path.contains_points(xys))[0] 
        return self.sub_indices[selection]
        
    def onselect(self, verts):
        print('Storing path')
        #Re add first point so path closes properly
        verts = np.array(verts)
        self.path = mplPath(np.concatenate([verts, verts[0][None,:]], axis=0)) 
        
        xys = self.collection.get_offsets()
        #indices of points in path
        self.selection = np.flatnonzero(self.path.contains_points(xys))
        self.master.set_selection(self.sub_indices[self.selection])
        
        self.apply_color()
        
    def set_selection(self, indices):
        #Called by master to sync highlighted indices between plots
        self.selection = np.flatnonzero(np.isin(self.sub_indices, indices))
        self.apply_color()
        
        