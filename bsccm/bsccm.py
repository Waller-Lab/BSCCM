import pandas as pd
import numpy as np
import zarr
import os
import json
import warnings
import requests
import tarfile


def download_data(mnist=True, coherent=False, tiny=False):
    """
    Download one of the 6 possible versions of BSCCM dataset
    
    mnist: download BSCCMNIST (downsized and downsampled version of BSCCM)
    coherent: download BSCCM-coherent or BSCCM-coherent-tiny
    tiny: the tiny version or the full version
    """


    location = '/home/hpinkard_waller/2tb_ssd/'
    doi_url = 'doi%3A10.5061%2Fdryad.9pg8d'
    version_index = -1
    file_index = 1

    # Get the version ID of the dataset
    api_url = "https://datadryad.org/api/v2/"
    versions = requests.get(api_url + 'datasets/{}/versions'.format(doi_url))
    version_id = versions.json()['_embedded']['stash:versions'][version_index]['_links']['self']['href'].split('/')[version_index]

    # Get the URL to download one particular file
    file = requests.get(api_url + 'versions/' + version_id + '/files').json()['_embedded']['stash:files'][file_index]
    file_name = file['path']
    download_url = 'https://datadryad.org' + file['_links']['stash:download']['href']

    # Download in chunks (so that really big files can be downloaded)
    chunk_size = 1024 * 1024 * 8
    iters = file['size'] / chunk_size
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(location + file_name, 'wb') as f:
            for i, chunk in enumerate(r.iter_content(chunk_size=chunk_size)): 
                print('Downloading {}, {:.1f}%\r'.format(file_name, 100 * i / iters ), end='')
                f.write(chunk)
    print('Finished downloading')


    loc = location + file_name[:-7] #remove .tar.gz
    print('Extracting to  {}...'.format(loc))
    file = tarfile.open(location + file_name)
    file.extractall(loc)
    file.close()
    print('Cleaning up')
    os.remove(location + file_name)
    print('Complete')



class BSCCM:

    def __init__(self, data_root, cache_index=False):
        """
        
        data_root: path to the top-level BSCCM directory
        cache_index: load the full index into memory. Set to true for increased performance at the expense of memory usage
        """
        self.global_metadata = json.loads(open(data_root + 'BSCCM_global_metadata.json').read())  
        print('Opening {}'.format(str(self)))
        if data_root[-1] != os.sep:
            data_root += os.sep
        self.data_root = data_root
        self.zarr_dataset = zarr.open(data_root + 'BSCCM_images.zarr', 'r')
        self.index_dataframe = pd.read_csv(data_root + 'BSCCM_index.csv', low_memory=not cache_index, index_col='global_index')
        self.size = len(self.index_dataframe)
        self.fluor_channel_names = self.global_metadata['fluorescence']['channel_names']
        self.led_array_channel_names = self.global_metadata['led_array']['channel_names']
        if 'BSCCM_surface_markers.csv' in os.listdir(data_root):
            self.surface_marker_dataframe = pd.read_csv(data_root + 'BSCCM_surface_markers.csv', index_col='global_index')
        if 'BSCCM_backgrounds.zarr' in os.listdir(data_root):
            self.backgrounds_and_shading = zarr.open(data_root + 'BSCCM_backgrounds.zarr', 'r')
        print('Opened {}'.format(str(self)))

        #TODO: add width and height
        
        #TODO: read specific name from global metadata
        
    def __str__(self):
        return self.global_metadata['name']
    
    def __repr__(self):
        return str(self)
        
    def read_image(self, index, channel, copy=False, convert_histology_rgb32=True):
        """
        
        TODO: add a note about how histology is translated on the fly to RGB32
        
        contrast_type: 'led_array', 'fluor', 'dpc', 'histology'
        """
        if index not in self.index_dataframe.index:
            raise Exception('{} is not a valid index into this dataset. Try using .get_indices to find a valid index'.format(index))

        # infer contrast type from channel
        if channel in self.global_metadata['led_array']['channel_names']:
            contrast_type = 'led_array'
        elif channel in self.global_metadata['fluorescence']['channel_names']:
            contrast_type = 'fluor'
        elif channel == 'histology':
            contrast_type = 'histology'
        elif channel == 'dpc' or channel == 'DPC':
            contrast_type = 'dpc'
        else:
            raise Exception('unrecognized channel: {}'.format(channel))

        entry = self.index_dataframe.loc[index]
        base_path = entry['data_path'] + '/'
        if contrast_type == 'led_array':
            base_path += contrast_type
            channel_index = self.led_array_channel_names.index(channel)
        elif contrast_type == 'fluor':
            base_path += contrast_type
            channel_index = self.fluor_channel_names.index(channel)
        elif contrast_type == 'dpc':
            base_path += contrast_type
            channel_index = None
        elif contrast_type == 'histology' :
            base_path += contrast_type
            channel_index = None
        else:
            raise Exception('unrecognized contrast_type')
        
        image = self.zarr_dataset[base_path + '/cell_{}'.format(index)]
        if channel_index is not None:
            image = image[channel_index]
        
        if contrast_type == 'histology' and convert_histology_rgb32 and 'MNIST' not in self.global_metadata['name']:
            image = np.array(image) / (2 ** 12)
            
        if copy:
            return np.array(np.squeeze(image))
        return np.squeeze(image)
        
    def get_indices(self, batch=None, slide_replicate=None, antibodies=None, 
                    has_matched_histology=False, shuffle=False, seed=None):
        sub_data_frame = self.index_dataframe
        if batch is not None:
            sub_data_frame = sub_data_frame[sub_data_frame.batch == batch]
        if slide_replicate is not None:
            sub_data_frame = sub_data_frame[sub_data_frame.slide_replicate == slide_replicate]
        if antibodies is not None:
            if type(antibodies) == list or type(antibodies) == tuple:
                sub_data_frame = sub_data_frame[sub_data_frame.antibodies.isin(antibodies)]    
            else:
                sub_data_frame = sub_data_frame[sub_data_frame.antibodies == antibodies]
        if has_matched_histology:
            sub_data_frame = sub_data_frame[sub_data_frame.has_matched_histology_cell]
        
        indices = sub_data_frame.index.to_numpy()
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(indices)
        return indices
    
    def get_corrected_fluor(self, indices):
        names = [
            'Fluor_690-_shading_corrected',   
            'Fluor_627-673_shading_corrected', 
            'Fluor_585-625_shading_corrected',             
            'Fluor_550-570_shading_corrected', 
            'Fluor_500-550_shading_corrected', 
            'Fluor_426-446_shading_corrected',
               ]
        return self.surface_marker_dataframe.loc[indices][names].to_numpy()
    
    def get_surface_marker_data(self, indices):
        four_spectra_model_names = [
           'CD123/HLA-DR/CD14_full_model_unmixed',
           'CD3/CD19/CD56_full_model_unmixed', 
           'CD45_full_model_unmixed',
           'CD16_full_model_unmixed']
            
        two_spectra_model_names = [
                           'CD45_single_antibody_model_unmixed',
                           'autofluor_single_antibody_model_unmixed',
                           'CD123_single_antibody_model_unmixed',
                           'CD19_single_antibody_model_unmixed',
                           'CD56_single_antibody_model_unmixed',
                           'CD14_single_antibody_model_unmixed',
                           'CD16_single_antibody_model_unmixed',
                           'HLA-DR_single_antibody_model_unmixed',
                           'CD3_single_antibody_model_unmixed',
                          ]
        
        four_spectra_data = self.surface_marker_dataframe.loc[indices][four_spectra_model_names].to_numpy()
        two_spectra_data = self.surface_marker_dataframe.loc[indices][two_spectra_model_names].to_numpy()
        return two_spectra_model_names, two_spectra_data, four_spectra_model_names, four_spectra_data
    
    def get_background(self, index, channel='Brightfield', percentile=50, dim=128):
        """
        Return a pixel-wise percentile over many images from the cell's location in the original data,
        specific to each physical slide
        
        contrast_type: 'led_array', 'fluor', 'dpc', 'histology'
        percentile: 5, 10, 20, 40, 50 (median) 
        """
        if 'MNIST' in str(self) or 'tiny' in str(self):
            warnings.warn('Backgrounds not included in {}'.format(str(self)))
        
        if percentile not in [5, 10, 20, 40, 50]:
            raise Exception('percentile must be one of: [5, 10, 20, 40, 50]')
        
        image = self.backgrounds_and_shading[channel]['{}_percentile'.format(percentile)]
  
        x, y = self.index_dataframe.loc[index].position_in_fov_x_pix, self.index_dataframe.loc[index].position_in_fov_y_pix
        cropped = image[..., int(y) - dim // 2: int(y) + dim // 2, int(x) - dim // 2: int(x) + dim // 2]
        return cropped
    
    