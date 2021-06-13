import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import zarr
import matplotlib.gridspec as gridspec
import os
import json

class BSCCM:

    def __init__(self, data_root):
        self.zarr_dataset = zarr.open(data_root + 'BSCCM_images_dpc32.zarr', 'r')
        self.index_dataframe = pd.read_csv(data_root + 'BSCCM_index.csv', low_memory=False)
        self.global_metadata = json.loads(open(data_root + 'BSCCM_global_metadata.json').read())
        self.size = len(self.index_dataframe)
        if 'BSCCM_surface_markers.csv' in os.listdir(data_root):
            self.surface_marker_dataframe = pd.read_csv(data_root + 'BSCCM_surface_markers.csv')
        if 'BSCCM_backgrounds_and_shading.zarr' in os.listdir(data_root):
            self.backgrounds_and_shading = zarr.open(data_root + 'BSCCM_backgrounds_and_shading.zarr', 'r')

    
    def read_image(self, index, contrast_type, channel=None, copy=False):
        """
        contrast_type: 'led_array', 'fluor', 'dpc', 'histology'
        """
        
        entry = self.index_dataframe.iloc[index]
        base_path = entry['data_path'] + '/'
        if contrast_type == 'led_array':
            base_path += contrast_type
            channel_index = self.global_metadata['led_array']['channel_names'].index(channel)
        elif contrast_type == 'fluor':
            base_path += contrast_type
            channel_index = self.global_metadata['fluorescence']['channel_names'].index(channel)
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

        if copy:
            return np.array(image)
        return image
        
    def get_indices(self, batch=None, slide_replicate=None, marker=None, 
                    has_matched_histology=False, shuffle=False):
        sub_data_frame = self.index_dataframe
        if batch is not None:
            sub_data_frame = sub_data_frame[sub_data_frame.batch == batch]
        if slide_replicate is not None:
            sub_data_frame = sub_data_frame[sub_data_frame.slide_replicate == slide_replicate]
        if marker is not None:
            if type(marker) == list:
                sub_data_frame = sub_data_frame[sub_data_frame.marker.isin(marker)]    
            else:
                sub_data_frame = sub_data_frame[sub_data_frame.marker == marker]
        if has_matched_histology:
            sub_data_frame = sub_data_frame[sub_data_frame.matched_histology_cell]
        
        indices = sub_data_frame.index.to_numpy()
        if shuffle:
            np.random.shuffle(indices)
        return indices
    
    def get_raw_fluor(self, indices):
        """
        Get the raw fluorescence measurements summed over image as N x D matrix
        """
        channel_names = ['BV421', 'BV510', 'BV570', 'BV605', 'BV650', 'BV711']
        return self.index_dataframe.iloc[indices][channel_names].to_numpy()
    
    def get_background_fluor(self, index, channel, dim=128):
        """
        Return the background fluorescence of the imaging system, as measured by looking
        at a median image of a dataset with no antibody stains, after it has been corrected for its
        own slide-specific offset
        """
        corr_index = int(self.index_dataframe.fluor_shading_correction_index.iloc[index])
        all_corrections = self.backgrounds_and_shading['unstained_backgrounds'][str(corr_index)]
        image = all_corrections['{}_{}'.format(channel, 50)]
        
        return image
    
#         offset = np.median(image[:150, :150])
        
#         background_fluor = image - offset
        
#         x, y = self.index_dataframe.iloc[index].position_in_fov_x_pix, self.index_dataframe.iloc[index].position_in_fov_y_pix
#         cropped = background_fluor[..., int(y) - dim // 2: int(y) + dim // 2, int(x) - dim // 2: int(x) + dim // 2]
#         return cropped
        
    
    def get_dataset_fluor_offset(self, index, channel):
        """
        get the median pixel intensity in the top left corner of the FOV for this dataset
        where there should be no in focus fluorescnce detected. This should measure 
        a dataset specific intensity offset
        """
        corr_index = int(self.index_dataframe.fluor_shading_correction_index.iloc[index])

         # There is one of these per each slide
        path = 'per_slide_backgrounds/{}/{}/{}/{}_{}'.format(
            self.index_dataframe.iloc[index].antibodies, 
            self.index_dataframe.iloc[index].batch,
            self.index_dataframe.iloc[index].slide_replicate, 
            channel, 50)
        image = self.backgrounds_and_shading[path]
        
        return image
#         return np.median(image[:150, :150])
    
    def get_unstained_background(self, index, channel, dim=128, percentile=20):
        """
        Get the fluoresncene background signal from images that have no antibody stains
        Assume this is represented by 20th percentile
        """        
        corr_index = int(self.index_dataframe.fluor_shading_correction_index.iloc[index])
        all_corrections = self.backgrounds_and_shading['unstained_backgrounds'][str(corr_index)]
        image = all_corrections['{}_{}'.format(channel, percentile)]

        x, y = self.index_dataframe.iloc[index].position_in_fov_x_pix, self.index_dataframe.iloc[index].position_in_fov_y_pix
        background = image[..., int(y) - dim // 2: int(y) + dim // 2, int(x) - dim // 2: int(x) + dim // 2]
        return background
    
    def get_shading(self, index, contrast_type='led_array', channel='Brightfield', percentile=50, dim=128):
        """
        Return a pixel-wise percentile over many images from the cell's location in the original data,
        specific to each physical slide
        
        contrast_type: 'led_array', 'fluor', 'dpc', 'histology'
        percentile: 5, 10, 20, 40, 50 (median) 
        """
        
        if contrast_type == 'led_array':
            # There are three of those for the whole dataset, so figure out which one used
            corr_index = int(self.index_dataframe.fluor_shading_correction_index.iloc[index])            
            image = self.backgrounds_and_shading['led_array_channel_backgrounds/{}/{}_{}'.format(corr_index, channel_percentile)]
        elif contrast_type == 'fluor':
            
            # There is one of these per each slide
            path = 'per_slide_backgrounds/{}/{}/{}/{}_{}'.format(
                self.index_dataframe.iloc[index].antibodies, 
                self.index_dataframe.iloc[index].batch,
                self.index_dataframe.iloc[index].slide_replicate, 
                channel, percentile)
            image = self.backgrounds_and_shading[path]
        else:
            raise Exception('No background for contrast type {}'.format(contrast_type))
            
        x, y = self.index_dataframe.iloc[index].position_in_fov_x_pix, self.index_dataframe.iloc[index].position_in_fov_y_pix
        cropped = image[..., int(y) - dim // 2: int(y) + dim // 2, int(x) - dim // 2: int(x) + dim // 2]
        return cropped
    
    
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
            ax.imshow(image, cmap='inferno')
            ax.set_axis_off()
