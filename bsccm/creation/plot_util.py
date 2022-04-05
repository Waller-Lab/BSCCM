#convenience code for plot export

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cmocean
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from matplotlib import cm


# plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
# mpl.rcParams['axes.spines.left'] = True
# mpl.rcParams['axes.spines.bottom'] = True
# mpl.rcParams['axes.spines.top'] = False
# mpl.rcParams['axes.spines.right']  = False

#editable text in fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#make text on figures look good
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('font', sansserif='Arial')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.style'] = 'normal'
# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "sans-serif"

plt.rc('axes', linewidth=2)    
plt.rc('xtick.major', width=2)    
plt.rc('ytick.major', width=2)    

plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    

##### General stylizing of plots #####
def default_format(ax):
    decimal_format_ticks(ax)
    sparse_ticks(ax)
    clear_spines(ax)
    zero_lims(ax)

def clear_spines(ax):
    ax.spines[["top", "right"]].set_visible(False)
    
def decimal_format_ticks(ax):
    def formatter(x, pos):
        if x == int(x):
            return str(int(x))
        return '{:.1f}'.format(x)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
def sparse_ticks(ax, mode='both'):
    if mode == 'both':
        ax.set_yticks([0, ax.get_yticks()[-1]])
        ax.set_xticks([0, ax.get_xticks()[-1]])
    elif mode == 'x':
        ax.set_xticks([0, ax.get_xticks()[-1]])
    elif mode == 'y':
        ax.set_yticks([0, ax.get_yticks()[-1]])
        
def zero_lims(ax, mode='both'):
    if mode == 'both':
        ax.set_xlim([0, ax.get_xlim()[-1]])
        ax.set_ylim([0, ax.get_ylim()[-1]])
    elif mode == 'x':
        ax.set_xlim([0, ax.get_xlim()[-1]])
    elif mode == 'y':
        ax.set_ylim([0, ax.get_ylim()[-1]])
        
    
###################################
    
    
    
    
    
    
def show_complex_image(image, ax):
    converted = cmocean.cm.phase_r((np.pi + np.angle(image)) / (np.pi * 2), alpha=np.abs(image) / np.max(np.abs(image)) )
    black = np.zeros_like(converted)
    black[..., -1] = 1
    ax.imshow(black)
    ax.imshow(converted)
    ax.set_axis_off()


def line_plot_with_phase(line_profile, ax, width=2):
    points = np.abs(np.array([np.arange(line_profile.size),
                       line_profile]).T.reshape(-1, 1, 2))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(-np.pi, np.pi)
    lc = LineCollection(segments, cmap=cmocean.cm.phase_r, norm=norm)
    # Set the values used for colormapping
    colors = np.concatenate([(np.angle(line_profile[:-1]) +
                              np.angle(line_profile[1:])) / 2,
                             np.angle(line_profile[-1])[None]])
    lc.set_array(colors)
    lc.set_linewidth(width)
    line = ax.add_collection(lc)
    ax.set_xlim(0, line_profile.size)
    ax.set_ylim(0, 1.1* np.max(np.abs(line_profile)))
    ax.set_yticks([0, np.max(np.abs(line_profile))])


def show_image(image, ax, contrast_max=None, contrast_min=0, name='', colorbar=True):
    if contrast_min is None:
        contrast_min = np.min(image)
    if contrast_max is None:
        contrast_max = np.max(image)
    im = ax.imshow(image, cmap='inferno', vmin=contrast_min, vmax=contrast_max)
    ax.set_axis_off()
    ax.set_title(name)
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_yticks([contrast_min, contrast_max])
    return im
    
def show_histogram(ax, data, bins, name):
    im = ax.hist(np.ravel(data), bins, density=True)
    plt.ylabel('Probability')
    ax.set_title(name)
    

def show_phase_colorbar(ax, max_amplitude=1):
    high = cmocean.cm.phase_r(np.linspace(0,1,400))[..., :3]
    shading = np.linspace(0,1,120)
    colorbar = np.swapaxes((shading[:, None, None] * high[None]), 0, 1)
    ax.imshow(colorbar, origin='lower')
    ax.set_xticks([0, shading.size])
    ax.set_xticklabels([0, max_amplitude])
    ax.set_xlabel('Amplitude')
    ax.set_yticks([0, high.shape[0]])
    ax.set_yticklabels(['0', '2$\pi$'])
    ax.set_ylabel('Phase')

    
def show_intensity_colorbar(ax, image):
    cmap_image = np.stack(20 *[cm.inferno(np.linspace(0,1,100))], axis=1)
    ax.imshow(cmap_image,  origin='lower')
    ax.set_xticks([])
    # ax.set_xticklabels([0, max_amplitude])
    # ax.set_xlabel('Amplitude')
    ax.set_yticks([0, cmap_image.shape[0]])
    ax.set_yticklabels([0, np.round(np.max(image))])
    ax.set_ylabel('Intensity')
