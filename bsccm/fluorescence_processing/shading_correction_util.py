import pandas as pd

import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from scipy import stats
from scipy.spatial import distance
import numpy as np
from scipy import interpolate
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
from bsccm import BSCCM

def lowess_2d(y_pos, x_pos, z_val, yx_query_points,
              alpha=0.1, weight_fn='tricubic', window_sigma=100):
   
    distances = np.linalg.norm(yx_query_points[:, None] - 
                       np.stack([y_pos, x_pos], axis=1)[None], axis=2)

    # find set of closest points to each query point
    closest_indices = np.argsort(distances, axis=1)[..., :int(alpha * y_pos.size)]
    closest_coords = np.stack([y_pos[closest_indices], x_pos[closest_indices]], axis=2)
    closest_targets = z_val[closest_indices]

    # Compute weighting function based on distance from center of window
    weight_distances = np.sqrt(np.sum((closest_coords - yx_query_points[:, None]) ** 2, axis=2))
    
    #normalize to 1
    weight_distances /= np.max(weight_distances)
    if weight_fn == 'tricubic':
        weights = (1 - weight_distances ** 3) ** 3
    elif weight_fn == 'gaussian':
        weights = np.exp( -0.5*(weight_distances / window_sigma) ** 2 )
    else:
        raise Exception('unknown weight fn')
    weights = np.sqrt(weights)

    #solve least squares problems and make predicitions
    predictions = []
    for query_point, w, X, y in zip(
        yx_query_points, weights, closest_coords, closest_targets):
        reg = LinearRegression().fit(X, y, w)
        predictions.append(reg.predict(query_point[None]))
    return np.array(predictions)


def plot_gridded_lowess(y_pos, x_pos, fluor, ax, N=20, alpha=None, weight_fn='tricubic',
                       window_sigma=100):
    #do LOWESS on a grid
    xx, yy = np.meshgrid(np.linspace(0, 2056, N), np.linspace(0, 2056, N))
    yx_query_points = np.stack([yy, xx], axis=-1).reshape([-1, 2])

    # compute alpha adaptively so it corresponds to same number cells
    if alpha is None:
        alpha = 2500 / x_pos.shape[0] 
        print('                     {:.2f} \r'.format(alpha), end='')
    predictions = lowess_2d(y_pos, x_pos, fluor, 
              yx_query_points, alpha=alpha, weight_fn=weight_fn, window_sigma=window_sigma)
    predictions = predictions.reshape([N, N])

    #mask out areas with nothing
    count = stats.binned_statistic_2d(y_pos, x_pos, fluor, bins=N, statistic='count').statistic
    predictions[count == 0] = np.nan
    
    im = ax.imshow(predictions, cmap='inferno')
    plt.colorbar(im, ax=ax, aspect=8)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
def show_spatial_histograms(bsccm, channel_names, figsize=(9,7), N=25, batch=0):
    replicate = 0
    antibodies_list = bsccm.index_dataframe['antibodies'].unique()

    fig, ax = plt.subplots(len(antibodies_list), len(channel_names), figsize=figsize)
    for i, antibodies in enumerate(antibodies_list):
        mask = np.logical_and(np.logical_and(bsccm.index_dataframe['antibodies'] == antibodies,
                                            bsccm.index_dataframe['batch'] == batch), 
                                            bsccm.index_dataframe['slide_replicate'] == replicate)
        mask_indices = np.flatnonzero(mask)

        for j, channel in enumerate(channel_names):


            if j == 0:
                ax[i, j].set_ylabel(antibodies)
            y_pos = bsccm.index_dataframe.loc[mask_indices, 'position_in_fov_y_pix'].to_numpy()
            x_pos = bsccm.index_dataframe.loc[mask_indices, 'position_in_fov_x_pix'].to_numpy()
            fluor = bsccm.surface_marker_dataframe.loc[mask_indices, channel].to_numpy()
            stat = stats.binned_statistic_2d(y_pos, x_pos, fluor, bins=N, 
                                             statistic=
                                            'mean'
#                                                  lambda x: np.percentile(x, 50)
                                            ).statistic

            contrast_max = np.nanpercentile(stat, 95)
            im = ax[i, j].imshow(stat, cmap='inferno', vmax=contrast_max, interpolation='nearest')

            plt.colorbar(im, aspect=8, ax=ax[i, j])
            ax[i, j].set_yticklabels([])
            ax[i, j].set_xticklabels([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xticks([])
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)
#             ax[i, j].set_title(channel)
    fig.suptitle('Batch {}'.format(batch))