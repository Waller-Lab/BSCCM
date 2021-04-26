import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
from pathlib import Path
import zarr
import matplotlib.gridspec as gridspec
from skimage.transform import rescale, resize, downscale_local_mean
import os


######
# TODO: change version of dataframe that youre using??
#####

home = str(Path.home())
data_root = '/home/henry/leukosight_data/crops_single_final/'
dataset = zarr.open(data_root + 'crops_single.zarr', 'r')
# rgb_dataset = zarr.open(data_root + 'rgb_crops.zarr', 'r')
csv_path = data_root + 'crops_fieldstop_filtered_cleaned_record_single.csv'
dataframe = pd.read_csv(csv_path, index_col=0, low_memory=False)
# rgb_dataframe = pd.read_csv(data_root + 'crops_record_histology_matched.csv', index_col=0, low_memory=False)
channel_names = dataset[list(dataset)[0]]['all_blobs'].attrs['channel_names']

all_filenames =  ['{}/{}'.format(int(dataset_index), crop_name) for dataset_index, crop_name 
         in zip(dataframe['dataset_index'].to_numpy(), dataframe['blob_name'].tolist())]

new_name = 'BSBCM_coherent.zarr'

channel_names = dataset['28']['all_blobs'].attrs['channel_names']
led_array_channel_names = [name for name in channel_names if 'led' in name]
fluor_channel_names = [name for name in channel_names if 'BV' in name]
led_array_channel_indices = np.flatnonzero(['led' in name for name in channel_names])
fluor_channel_indices = np.flatnonzero(['BV' in name for name in channel_names])


new_file = zarr.open(data_root + new_name, mode='w')
new_file.attrs['fluorescence_channels'] = fluor_channel_names
new_file.attrs['led_array_channel_names'] = led_array_channel_names

for i in range(len(dataframe)):
    print(str(i) + ' of  {}\r'.format(len(dataframe)), end='')

    entry = dataframe.iloc[i]

    marker = 'all' if (entry.dataset_index == 28) or (entry.dataset_index == 29) else 'unstained'
    
    replicate = 1
    base_path = 'batch_{}/{}/replicate_{}'.format(2, marker, replicate)
    fluor_dest = base_path + '/fluor/cell_{}'.format(i)
    lf_dest = base_path + '/led_array/cell_{}'.format(i)
    dpc_dest = base_path + '/dpc/cell_{}'.format(i)

    lf_source = str(int(entry['dataset_index'])) + '/' + str(entry['blob_name'])    

    fluor_data = dataset[lf_source][int(np.min(fluor_channel_indices)) : int(np.max(fluor_channel_indices)+1)] 
    led_array_data = dataset[lf_source][int(np.min(led_array_channel_indices)) : int(np.max(led_array_channel_indices)+1)]
    dpc_data = dataset[lf_source][-1][None, ...]

    new_file.create_dataset(name=fluor_dest, data=fluor_data.astype(np.uint16), chunks=(1,150,150))
    new_file.create_dataset(name=lf_dest, data=led_array_data.astype(np.uint16), chunks=(1,150,150))
    new_file.create_dataset(name=dpc_dest, data=dpc_data, chunks=(1,150,150))