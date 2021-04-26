"""
Take in raw fluorescence images and compute a scalar fluorescence score
"""

import zarr
import dask.array as da
import pandas as pd
import numpy as np
from scipy import stats, ndimage

from pathlib import Path

home = str(Path.home())
data_root = home + '/leukosight_data/crops_fluorescence_data_and_record/'

# Which dataset indices correspond to which flatfield
# TODO Replace this with something read from the dataframe
# TODO Need to account for dataset 13 being off...
# datasets_813 = [0, 1, 2, 3, 4, 5, 7]
# datasets_828 = [8, 9, 11, 12, 13, 15, 16]
# datasets_920 = [17, 18, 19, 20, 21, 22, 23, 24, 26, 27]

datasets_813 = []
datasets_828 = []
datasets_920 = [28, 29, 30, 31]

shading_file = np.load(file=data_root + 'shading_corrections.npz')
dataframe = pd.read_csv(data_root + 'crops_single_final/crops_fieldstop_filtered_cleaned_record_single.csv')
crops_dataset = zarr.open(data_root + 'crops_single_final/crops_single.zarr' , mode='r')

fluor_channel_indices = np.array([crops_dataset[str(datasets_920[0])]['all_blobs'].attrs['channel_names'].index(n) for n in
                   ['F1_BV711', 'F2_BV650', 'F3_BV605', 'F4_BV570', 'F5_BV510', 'F6_BV421']])

fluor_names = ['BV711', 'BV650', 'BV605', 'BV570', 'BV510', 'BV421']
new_cols = {name: np.zeros((dataframe.size, )) for name in fluor_names + [n + '_without_local_background_sub' for n in fluor_names]}
#add new columns to dataframe
dataframe = dataframe.join(pd.DataFrame(new_cols))

def crop(img, center, dim=150):
    return img[..., int(center[0]) - dim // 2: int(center[0]) + dim // 2,
               int(center[1]) - dim // 2: int(center[1]) + dim // 2]

def compute_fluoresence(dataframe, dataset_index, background, shading=None):
    """
    Do background substraction and optionally, shading
    """
    print('dataset {}'.format(dataset_index))
    in_dataset_indices = np.flatnonzero(dataframe['dataset_index'] == dataset_index)
    dataset_dataframe = dataframe.loc[in_dataset_indices]
    y_coords = dataset_dataframe['blob_y'].to_numpy()
    x_coords = dataset_dataframe['blob_x'].to_numpy()
    crop_names = dataset_dataframe['blob_name'].to_list()
    if dataset_index == 13:
        x_coords -= 196  # wrong ROI on this dataset
    mean_fluors_without_background_sub = []
    mean_fluors = []
    for i, (y, x, crop_name) in enumerate(zip(y_coords, x_coords, crop_names)):
        print('{} of {}\r'.format(i, y_coords.size), end='')
        background_crop = crop(background, (y, x))
        if shading is not None:
            shading_crop = crop(shading, (y, x))
        else:
            shading_crop = np.ones_like(background_crop)
        fluor_crop = crops_dataset['{}/{}'.format(dataset_index, crop_name)].get_orthogonal_selection(
            (fluor_channel_indices, slice(None), slice(None)))
        fluor_crop_corrected = (fluor_crop - background_crop) / shading_crop
        # mask only pixels within 130 pixel diameter circle
        yy, xx = np.meshgrid(np.arange(150), np.arange(150))
        fluor_mask = np.sqrt((yy - 75) ** 2 + (xx - 75) ** 2) < 65
        fluor_mask_pixels = fluor_crop_corrected[:, fluor_mask]
        local_background = np.percentile(fluor_mask_pixels, 5, axis=1)
        mean_fluor = np.sum(fluor_mask_pixels - local_background[:, None], axis=1)
        mean_fluor_without_local_subtraction = np.sum(fluor_mask_pixels, axis=1)

        # napari_show({'b': background_crop, 's': shading_crop, 'f': fluor_crop, 'f-b': fluor_crop - background_crop,
        #              'corrected': fluor_crop_corrected})

        mean_fluor[mean_fluor < 0] = 0
        mean_fluor_without_local_subtraction[mean_fluor_without_local_subtraction < 0]
        mean_fluors.append(mean_fluor)
        mean_fluors_without_background_sub.append(mean_fluor_without_local_subtraction)

    without_local_sub = np.stack(mean_fluors_without_background_sub)
    mean_fluors = np.stack(mean_fluors)
    return mean_fluors, without_local_sub

def correct_and_compute_fluorescence(ids, do_local_backgrounds=True, shading=True):
    # Compute shading corrections from loess of all stain data
    mean_fluorescences = {}
    without_background_subtraction = {}
    for dataset in ids:
        if dataset in datasets_813:
            background = shading_file['813_backgrounds']
            ff = shading_file['813_ff_slide']
        elif dataset in datasets_828:
            background = shading_file['828_backgrounds']
            ff = shading_file['828_ff_slide']
        elif dataset in datasets_920:
            background = shading_file['920_backgrounds']
            ff = shading_file['920_ff_slide']
        background = background * 0.92
        shading_corr = ff - 0.92 * background

        #normalize by median brightness
        shading_corr = shading_corr / np.mean(shading_corr, axis=(1,2))[:,None,None]

        mean_fluorescence, fluor_without_local_background = compute_fluoresence(dataframe, dataset, background,
                                    shading=shading_corr if shading else None)
        mean_fluorescences[dataset] = mean_fluorescence
        without_background_subtraction[dataset] = fluor_without_local_background
    return mean_fluorescences, without_background_subtraction


ids = np.unique(dataframe['dataset_index'].to_numpy().astype(np.int))

mean_fluorescences, mean_fluorescences_no_bckd_sub = correct_and_compute_fluorescence(ids, do_local_backgrounds=True, shading=True)
for dataset_index in ids:
    in_dataset_indices = np.flatnonzero(dataframe['dataset_index'] == dataset_index)
    new_data = np.concatenate([mean_fluorescences[dataset_index],
                               mean_fluorescences_no_bckd_sub[dataset_index]], axis=1)
    for i, col_name in enumerate(new_cols.keys()):
        dataframe.loc[in_dataset_indices, col_name] = new_data[:, i]


dataframe.to_csv(data_root + 'crops_filtered_record_single_with_fluor_normalized_ff.csv', index=False)
