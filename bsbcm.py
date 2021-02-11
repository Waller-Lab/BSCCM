import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import zarr
import matplotlib.gridspec as gridspec
import os

class BSBCM:

    def __init__(self, data_root):
        self.zarr_dataset = zarr.open(data_root + 'BSBCM.zarr', 'r')
        self.dataframe = pd.read_csv(data_root + 'BSBCM.csv', low_memory=False)
        self.led_array_channel_names = self.zarr_dataset.attrs['led_array_channel_names']
        self.fluor_channel_names = self.zarr_dataset.attrs['fluorescence_channels']
        self.size = len(self.dataframe)
    
    def read_image(self, index, contrast_type, channel=None, copy=False):
        """
        contrast_type: 'led_array', 'fluor', 'dpc', 'histology'
        """
        
        entry = self.dataframe.iloc[index]
        base_path = entry['base_path']
        if contrast_type == 'led_array':
            base_path += contrast_type
            channel_index = self.led_array_channel_names.index(channel)
        elif contrast_type == 'fluor':
            base_path += contrast_type
            channel_index = self.fluor_channel_names.index(channel)
        elif contrast_type == 'dpc':
            base_path += contrast_type
            channel_index = 0
        elif contrast_type == 'histology':
            base_path += contrast_type
            channel_index = None
        else:
            raise Exception('unrecognized contrast_type')
        
        image = self.zarr_dataset[base_path + '/cell_{}'.format(index)]
        if channel_index is not None:
            image = image[channel_index]
        if contrast_type == 'histology':
            image = np.moveaxis(image, 0, -1)

        if copy:
            return np.array(image)
        return image
        
    def has_histology_mask(self):
        return self.dataframe['matched_histology_cell'].to_numpy()
    
    def plot_montage(self, indices, contrast_type='dpc', channel=None,  size=(10, 10)):
        """
        Read images in the list of indices and plot as a montage
        """
        dim_size = int(np.sqrt(len(indices)))
        fig = plt.figure(figsize=size)
        gs1 = gridspec.GridSpec(dim_size, dim_size)
        gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

        images = np.stack([self.read_image(index=i, contrast_type=contrast_type, channel=channel)
                  for i in indices], axis=0)
        if contrast_type == 'histology':
            #mpl doesnt show 16 bit rgb so convert to float
            images = images / (2 ** 12) #12 bit image
        for index, image in enumerate(images):
            ax = plt.subplot(gs1[index])
            ax.imshow(image)
            ax.set_axis_off()
            
