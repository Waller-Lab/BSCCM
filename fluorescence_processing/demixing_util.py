import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit

from jax import grad, value_and_grad
import jax.numpy as jnp
import os

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import matplotlib.image as mplimg
from IPython.display import display




def compute_spectra(bsccm, channel_names, unmixed_channel_names, single_markers, batch):
    """
    Compute the spectra and brightness of those that will be used for unmixing
    """
    single_marker_unmix_channel_spectra = {}
    single_marker_unmix_channel_brightness = {}
    unmix_channel_spectra = {}
    unmix_channel_brightness = {}
    for name in unmixed_channel_names:
        single_unmix_spectra = []
        for m in single_markers:
            if m in name:
                selection_name = 'selection_example_{}_positive_cells_batch_{}'.format(m, batch)
                spectrum = get_fluor_spectrum(selection_name, bsccm, channel_names)
                single_marker_unmix_channel_spectra[m] = spectrum / np.linalg.norm(spectrum)
                single_marker_unmix_channel_brightness[m] = np.linalg.norm(spectrum)
                single_unmix_spectra.append(spectrum)

        mean_spectra = np.mean(np.stack(single_unmix_spectra, axis=0), axis=0)
        brightness = np.linalg.norm(mean_spectra)
        unmix_channel_spectra[name] = mean_spectra / brightness
        unmix_channel_brightness[name] = brightness
        
    return single_marker_unmix_channel_spectra, single_marker_unmix_channel_brightness, unmix_channel_spectra, unmix_channel_brightness
    

def get_fluor_spectrum(name, bsccm, channel_names):
    """
    Get the spectrum of a particular fluor by looking at difference in median
    between marked and unmarked populations
    """
    marked_indices, non_marked_indices = get_marked_unmarked_indices(name, bsccm)
    non_marked_fluor =  bsccm.surface_marker_dataframe.loc[non_marked_indices, channel_names].to_numpy()
    marked_fluor = bsccm.surface_marker_dataframe.loc[marked_indices, channel_names].to_numpy()
    
#     marked_center = np.percentile(marked_fluor, 95, axis=0) 
#     non_marked_center = np.median(non_marked_fluor, axis=0)
    
    marked_center = np.mean(marked_fluor, axis=0) 
    non_marked_center = np.mean(non_marked_fluor, axis=0)
    
    spectrum =  marked_center - non_marked_center
    spectrum[spectrum < 0] = 0
    return spectrum

def load_mixed_data_subset(mixed_data, antibodies, selections, bsccm, batch=0):
    """
    Load the data of a single antibody (or none for autofluor), along with the mask
    that shows marked populations

    """
    indices = bsccm.get_indices(antibodies='unstained' if antibodies == 'autofluor' else antibodies, batch=batch)
    data = mixed_data[indices]
    #if it has a positively marked population
    name = 'selection_example_{}_positive_cells_batch_{}'.format(antibodies, batch)
    if name in selections[batch]:
        marked_mask = get_marked_mask(name, bsccm)
    else:
        marked_mask = None
    return data, marked_mask

def run_experiment(spectra, spectra_names, bsccm, mixed_data, stain_antibodies, selections, reweighting, l1_reg, scatter_fig, scatter_ax, callback_every=50, path='/home/henry/leukosight_data/demixing_results/'):
    """
    Run a single hyperparameter sweep
    """
    for a in scatter_ax:
        a.clear()
    show_antibodies = [m for m in spectra_names if stain_antibodies in m][0]

    data, mask = load_mixed_data_subset(mixed_data, stain_antibodies, selections, bsccm, batch=0)

    callback = lambda i, loss_history, unmixed: demix_plot_callback(i, loss_history, unmixed, spectra_names, show_antibodies, mask,
                        loss_fig=None, loss_ax=None,
                        scatter_fig=scatter_fig, scatter_ax=scatter_ax, plot_loss_history=500, log_plots=False)

    unmixed, background_spectrum = do_factorization(data, spectra,
                     l1_reg = l1_reg,
                        momentum=0.9,
                        learning_rate = 1e3,
                        background_learning_rate=1e-1,
                        reweighting=reweighting, 
                        callback_every=callback_every,
                        callback=callback)
    
    
    
    save_name = '{}__{}__{}'.format(stain_antibodies, l1_reg, '-'.join(['{:.2f}'.format(r) for r in reweighting]))
    plt.savefig('{}{}.png'.format(path, save_name), transparent=True)

def show_experiment_results(path = '/home/henry/leukosight_data/demixing_results/'):
    """
    Show interactive GUI for viewing the results of hyperparameter sweep experiments
    """
    files = {}
    ab_options = set()
    reg_options = set()
    reweight_options = set()

    for file in os.listdir(path):
        antibody, reg, reweight = file[:-4].split('__')
        reg = float(reg)
    #     reweight = (float(r) for r in reweight.split('-'))
        ab_options.add(antibody)
        reg_options.add(reg)
        reweight_options.add(reweight)
        if antibody not in files:
            files[antibody] = {}
        if reg not in files[antibody]:
            files[antibody][reg] = {}
        if reweight not in files[antibody][reg]:
            files[antibody][reg][reweight] = {}
        files[antibody][reg][reweight] = path + file

    antibody_dropdown = widgets.Dropdown(
            options=ab_options,
            value=list(ab_options)[0],
            description='Antibody:',
            disabled=False,
        )

    reg_dropdown = widgets.Dropdown(
            options=reg_options,
            value=list(reg_options)[0],
            description='Regularization:',
            disabled=False,
        )

    reweight_dropdown = widgets.Dropdown(
            options=reweight_options,
            value=list(reweight_options)[0],
            description='Reweight:',
            disabled=False,
        )

    fig, ax = plt.subplots(figsize=(15, 2.5))

    def dropdown_callback(widget):
        ax.clear()
        reweight_dropdown.options = list(files[antibody_dropdown.value][reg_dropdown.value].keys())
        if reweight_dropdown.value not in reweight_dropdown.options:
            reweight_dropdown.value = reweight_dropdown.options[0]
        try:
            filename = files[antibody_dropdown.value][reg_dropdown.value][reweight_dropdown.value]
            print(filename)
            img = mplimg.imread(filename)
            ax.imshow(img)
            ax.set_axis_off()
        except:
            pass #nothing with these combos


    antibody_dropdown.observe(dropdown_callback, names='value')    
    reg_dropdown.observe(dropdown_callback, names='value')    
    reweight_dropdown.observe(dropdown_callback, names='value')    

    display(widgets.HBox([antibody_dropdown, reg_dropdown, reweight_dropdown]))

    dropdown_callback(None)

def do_factorization(data, spectra, l1_reg = 1e0,
                    momentum=0.9,
                    learning_rate = 1e3,
                    background_learning_rate = 1e-1,
                    background_reg = 5e-2,
                    reweighting = jnp.ones(6), 
                     callback_every=50,
                     callback=None,
                    scatter_fig=None,
                    scatter_ax=None,
                    stopping_error=0.0005):
    """
    Do NNMF with given spectra a a fixed background
    """
    if data.size == 0:
        raise Exception('Empty data')

    def forward_model(unmixed, background_ones, background_spectrum, reweighting, spectra):    
        mixed = jnp.dot(jnp.concatenate([unmixed, background_ones], axis=1),
                        jnp.concatenate([spectra, background_spectrum], axis=0))

        loss = jnp.mean((data - mixed)**2)
        reweighted = jnp.concatenate([unmixed[:, i] * reweighting[i] for i in range(spectra.shape[0])], axis=-1)

        regularization = 0
        if l1_reg != 0:
            regularization += l1_reg * jnp.mean(jnp.abs(reweighted)) 
            regularization += background_reg * jnp.sum(background_spectrum ** 2)

        return loss + regularization 

    val_grad_fn = jit(value_and_grad(forward_model, [0, 2]))

    target = jnp.array(data)
    unmixed = jnp.ones((data.shape[0], spectra.shape[0]))
    background_ones = jnp.ones((unmixed.shape[0], 1))
    spectra = jnp.array(spectra)
    background_spectrum = jnp.ones([1, spectra.shape[1]]) * 1e-2

    loss_history = []
    step = 0
    for i in range(0, int(1e7)):
        loss, (unmixed_grad, background_spectrum_grad) = val_grad_fn(unmixed, background_ones, background_spectrum, reweighting, spectra)
        loss_history.append(loss)

        # take step
        step = momentum * step - learning_rate * unmixed_grad
        unmixed += step

        background_spectrum -= background_learning_rate * background_spectrum_grad 

        #enforce constraints
        unmixed = unmixed * jnp.logical_not(unmixed < 0)
        background_spectrum = background_spectrum * jnp.logical_not(background_spectrum < 0)

        if i > 50:
            rel_error = (np.abs(loss_history[-50] - loss_history[-1]) / loss_history[-1]) 
        else: 
            rel_error = 1
        print(('{}: \tloss: {:.3f}\trel_error: {:.4f}\t\t' + (int(background_spectrum.size)*'{:.1f}  ') + '\t\t\t\t\r').format(
            i, loss, rel_error, *jnp.ravel(background_spectrum)), end='')

#         if i == 50:
#             callback(i, loss_history, unmixed)
#             return unmixed, background_spectrum
        
        # stop when error reudction is less than 0.1 % in last 100 interations        
        if rel_error < stopping_error:
            if callback is not None:
                callback(i, loss_history, unmixed) # plot final result
            break 
        if i % callback_every == 0 and callback is not None:
            callback(i, loss_history, unmixed)
            
    return unmixed, background_spectrum


def demix_plot_callback(i, loss_history, unmixed, spectra_names, show_antibody, marked_mask=None, loss_fig=None, loss_ax=None, scatter_fig=None, scatter_ax=None, plot_loss_history=500, log_plots=False):
    marker_index = [show_antibody in m for m in spectra_names].index(True)
    if loss_fig is not None:
        loss_ax.clear()
        loss_ax.semilogy(range(i, i+plot_loss_history)[:i+1], loss_history[-plot_loss_history:])
        loss_fig.canvas.draw()

    other_marker_indices = list(range(len(spectra_names)))
    del other_marker_indices[marker_index]
    for j, other_index in enumerate(other_marker_indices):
        scatter_ax[j].clear()
        if log_plots:
            log_data_other = np.log(1e-2 + unmixed[:, other_index])
            log_data_marker = np.log(1e-2 + unmixed[:, marker_index])
            scatter_ax[j].scatter(log_data_other[np.logical_not(marked_mask)], 
                          log_data_marker[np.logical_not(marked_mask)], 
                s=1, c='k')
            scatter_ax[j].scatter(log_data_other[marked_mask], 
                          log_data_marker[marked_mask], 
                s=1, c='lime')
            #Give good axes in spite of one big outlier
            scatter_ax[j].set_xlim([np.min(log_data_other) -0.5, 1.2 * np.percentile(log_data_other, 99.9)])
            scatter_ax[j].set_ylim([np.min(log_data_marker) - 0.5, 1.2 * np.percentile(log_data_marker, 99.9)])
        else:
            scatter_ax[j ].scatter(
                unmixed[np.logical_not(marked_mask), other_index],
                unmixed[np.logical_not(marked_mask), marker_index], 
                s=1, c='k')
            scatter_ax[j ].scatter(
                unmixed[marked_mask, other_index], 
                 unmixed[marked_mask, marker_index], 
                s=1, c='lime')
            #Give good axes in spite of one big outlier
            scatter_ax[j].set_xlim([-1, 1.4 * np.percentile(unmixed[:, other_index], 99.8)])
            scatter_ax[j].set_ylim([-1, 1.4 * np.percentile(unmixed[:, marker_index], 99.8)])              

        scatter_ax[j].set_title(spectra_names[other_marker_indices[j]])
        if j == 0:                                 
            scatter_ax[j ].set_ylabel(show_antibody)
    scatter_ax[-1].clear()
    if log_plots:
        scatter_ax[-1].hist(np.log(1e-2 + unmixed[np.logical_not(marked_mask), marker_index]), 50, color='k', alpha=0.5, log=True)
        scatter_ax[-1].hist(np.log(1e-2 + unmixed[marked_mask, marker_index]), 50, color='lime', alpha=0.5, log=True)
    else:
        scatter_ax[-1].hist(unmixed[np.logical_not(marked_mask), marker_index], 50, color='k', alpha=0.5, log=True)
        scatter_ax[-1].hist(unmixed[marked_mask, marker_index], 50, color='lime', alpha=0.5, log=True)
        
    scatter_ax[-1].set_xlabel(show_antibody)
    
    scatter_fig.canvas.draw()

def get_marked_unmarked_indices(name, bsccm):
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

def get_marked_mask(name, bsccm):
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


def plot_spectral_vector_selection(name, bsccm, channel_names, display_ch_indices=(0,1)):
    """
    make a plot of marked and non-marked populations and vector difference
    """
    marked_indices, non_marked_indices = get_marked_unmarked_indices(name, bsccm, channel_names)
    non_marked_fluor =  bsccm.surface_marker_dataframe.loc[non_marked_indices, channel_names].to_numpy()
    marked_fluor = bsccm.surface_marker_dataframe.loc[marked_indices, channel_names].to_numpy()
    
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
