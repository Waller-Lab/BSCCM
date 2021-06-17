import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

def get_marked_unmarked_indices(name, bsccm, channels):
    """
    Get the fluorescence for the populations that have been marked and not marked
    """
    population = bsccm.surface_marker_dataframe[name]
    marked_indices = np.flatnonzero(population)


    #get all cells from same population of the marked cells
    antibodies = bsccm.index_dataframe.loc[marked_indices].antibodies.unique()[0]
    batch = bsccm.index_dataframe.loc[marked_indices].batch.unique()[0]
    slide_replicate = bsccm.index_dataframe.loc[marked_indices].slide_replicate.unique()[0]
    non_marked_indices = bsccm.get_indices(batch=batch, antibodies=antibodies, 
                                          slide_replicate=slide_replicate)

    #remove the marked ones
    mask = np.logical_not(np.isin(non_marked_indices, marked_indices))
    non_marked_indices = non_marked_indices[mask]
    return  marked_indices, non_marked_indices

def get_marked_mask(name, bsccm, channels):
    """
    get a mask for the marked cells within the subset it comes from 
    """
    population = bsccm.surface_marker_dataframe[name]
    marked_indices = np.flatnonzero(population)

    #get all cells from same population of the marked cells
    antibodies = bsccm.index_dataframe.loc[marked_indices].antibodies.unique()[0]
    batch = bsccm.index_dataframe.loc[marked_indices].batch.unique()[0]
    slide_replicate = bsccm.index_dataframe.loc[marked_indices].slide_replicate.unique()[0]
    non_marked_indices = bsccm.get_indices(batch=batch, antibodies=antibodies, 
                                          slide_replicate=slide_replicate)

    #remove the marked ones
    return np.isin(non_marked_indices, marked_indices)

def get_fluor_spectrum(name, bsccm, channels):
    """
    Get the spectrum of a particular fluor by looking at difference in median
    between marked and unmarked populations
    """
    marked_indices, non_marked_indices = get_marked_unmarked_indices(name, bsccm, channels)
    non_marked_fluor =  bsccm.surface_marker_dataframe.loc[non_marked_indices, channels].to_numpy()
    marked_fluor = bsccm.surface_marker_dataframe.loc[marked_indices, channels].to_numpy()
    

    marked_center = np.median(marked_fluor, axis=0) 
    non_marked_center = np.median(non_marked_fluor, axis=0)
    return marked_center - non_marked_center

def plot_spectral_vector_selection(name, bsccm, channels, display_ch_indices=(0,1)):
    """
    make a plot of marked and non-marked populations and vector difference
    """
    marked_indices, non_marked_indices = get_marked_unmarked_indices(name, bsccm, channels)
    non_marked_fluor =  bsccm.surface_marker_dataframe.loc[non_marked_indices, channels].to_numpy()
    marked_fluor = bsccm.surface_marker_dataframe.loc[marked_indices, channels].to_numpy()
    
    marked_center = np.median(marked_fluor, axis=0) 
    non_marked_center = np.median(non_marked_fluor, axis=0)

    plt.figure()
    plt.scatter(non_marked_fluor[:, display_ch_indices[0]],
                non_marked_fluor[:, display_ch_indices[1]], c ='k')
    plt.scatter(marked_fluor[:, display_ch_indices[0]],
                marked_fluor[:, display_ch_indices[1]], c='lime')

    plt.scatter(marked_center[display_ch_indices[0]], 
                marked_center[display_ch_indices[1]], c='red')
    plt.scatter(non_marked_center[display_ch_indices[0]],
                non_marked_center[display_ch_indices[1]], c='red')

    plt.plot([marked_center[display_ch_indices[0]], 
              non_marked_center[display_ch_indices[0]]], 
            [marked_center[display_ch_indices[1]],
             non_marked_center[display_ch_indices[1]]], 'r') 
