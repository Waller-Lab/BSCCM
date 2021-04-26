import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from pathlib import Path
import zarr
import matplotlib.gridspec as gridspec
from skimage.transform import rescale, resize, downscale_local_mean
import os

home = str(Path.home())
data_root = home + '/leukosight_data/'
dataset = zarr.open(data_root + 'crops_fieldstop_filtered.zarr', 'r')
rgb_dataset = zarr.open(data_root + 'rgb_crops.zarr', 'r')
csv_path = data_root + 'crops_record_with_batch_corrections.csv'
dataframe = pd.read_csv(csv_path, index_col=0, low_memory=False)
rgb_dataframe = pd.read_csv(data_root + 'crops_record_histology_matched.csv', index_col=0, low_memory=False)
channel_names = dataset[list(dataset)[0]]['all_blobs'].attrs['channel_names']

all_filenames =  ['{}/{}'.format(int(dataset_index), crop_name) for dataset_index, crop_name 
         in zip(dataframe['dataset_index'].to_numpy(), dataframe['blob_name'].tolist())]

new_name = 'BSBCM.zarr'

channel_names = dataset['0']['all_blobs'].attrs['channel_names']
led_array_channel_names = channel_names[:22] + [channel_names[-3]]
fluor_channel_names = channel_names[22:28]
new_file = zarr.open(data_root + new_name, mode='w')
new_file.attrs['fluorescence_channels'] = fluor_channel_names
new_file.attrs['led_array_channel_names'] = led_array_channel_names

for i in range(len(dataframe)):
    print(str(i) + ' of  {}\r'.format(len(dataframe)), end='')

    entry = dataframe.iloc[i]
    rgb_entry = rgb_dataframe.iloc[i]
    
    
    replicate = 2 if ((entry['dataset_index'] ==  24) or (entry['dataset_index'] ==  26)) else 1
    base_path = 'batch_{}/{}/replicate_{}'.format(int(entry['batch']), entry['marker'], replicate)
    fluor_dest = base_path + '/fluor/cell_{}'.format(i)
    lf_dest = base_path + '/led_array/cell_{}'.format(i)
    dpc_dest = base_path + '/dpc/cell_{}'.format(i)
    hist_dest = base_path + '/histology/cell_{}'.format(i)

    lf_source = str(int(entry['dataset_index'])) + '/' + str(entry['blob_name'])
    if rgb_entry['histology_match']:
        histology_source = rgb_entry.histology_dataset_name + '/' + rgb_entry.closest_histology_cell_name
    else:
        histology_source = None
    

    fluor_data = dataset[lf_source][22:28]
    led_array_data = np.concatenate([dataset[lf_source][:22], dataset[lf_source][-3][None, ...]])
    dpc_data = dataset[lf_source][-1][None, ...]

    new_file.create_dataset(name=fluor_dest, data=fluor_data.astype(np.uint16), chunks=(1,150,150))
    new_file.create_dataset(name=lf_dest, data=led_array_data.astype(np.uint16), chunks=(1,150,150))
    new_file.create_dataset(name=dpc_dest, data=dpc_data, chunks=(1,150,150))
    
    if histology_source is not None:
        hist_data = rgb_dataset[histology_source]
        new_file.create_dataset(name=hist_dest, data=hist_data, chunks=None)
