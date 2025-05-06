import pandas as pd
import numpy as np
import zarr
import os
import json
import warnings
import requests
import tarfile
import io
from tqdm import tqdm
import shutil 
import requests
from pathlib import Path



def download_dataset(location='.', coherent=False, tiny=True, mnist=False, token=None):
    """
    Downloads the BSCCM dataset to the specified location.
    If location is not specified, the current directory is used.

    Args:
        location (str): The location to download the dataset to.
        coherent (bool): Whether to download the coherent (single LED illumination patterns) version of the dataset.
        tiny (bool): Whether to download the tiny version of the dataset, a subsample of the full dataset.
        MNIST (bool): Whether to download the version of the dataset with MNIST sized images
        token: (Debugging only) for accessing versions of the dataset not yet released on Dryad.

    Returns:
        The path to the downloaded dataset.
    """

    # add trailing slash if not there
    if location[-1] != os.sep:
        location += os.sep

    dataset_name = 'BSCCM' if not mnist else 'BSCCMNIST'
    if coherent:
        dataset_name += '-coherent'
    if tiny:
        dataset_name += '-tiny'
    dataset_name += '.tar.gz'


    doi = 'doi%3A10.5061%2Fdryad.sxksn038s'
    base_url = "https://datadryad.org"

    # Set up the headers
    headers = { "Authorization": f"Bearer {token}"} if token is not None else None

    versions = requests.get(base_url + f'/api/v2/datasets/{doi}/versions', headers=headers)
    version_id = versions.json()['_embedded']['stash:versions'][-1]['_links']['self']['href'].split('/')[-1]

    # Function to get all files, handling pagination
    def get_all_files(version_id):
        all_files = []
        url = base_url + '/api/v2/versions/' + version_id + '/files'
        while url:
            print(f'Fetching file metadata {len(all_files)}...', end='\r')
            response = requests.get(url, headers=headers)

            # Check if the response status code indicates success
            if response.status_code != 200:
                print(f"Failed to fetch data: {response.status_code}")
                break

            # Try to decode JSON only if the response contains content
            if response.content:
                data = response.json()
                all_files.extend(data['_embedded']['stash:files'])
                links = data.get('_links', {})
                next_link = links.get('next', {}).get('href')

                if next_link:
                    if next_link.startswith('/'):
                        url = base_url + next_link
                    else:
                        url = next_link
                else:
                    url = None
            else:
                print("No content in response")
                break

        return all_files

    files = get_all_files(version_id)

    # find files relevant to this dataset
    files = [f for f in files if dataset_name in f['path']]

    # Filter out files that already exist with the correct size
    files_to_download = []
    skipped_size = 0
    for file_info in files:
        file_path = location + file_info['path']
        if os.path.exists(file_path) and os.path.getsize(file_path) == file_info['size']:
            print(f"Skipping {file_info['path']} - already exists with same size")
            skipped_size += file_info['size']
        else:
            files_to_download.append(file_info)

    download_chunk_size = 1024 * 1024 * 8  # 8 MB
    total_size = sum(f['size'] for f in files)

    # Adjust total size to reflect only files that need downloading
    actual_download_size = total_size - skipped_size
    print(f'Files to download: {len(files_to_download)} of {len(files)}')

    # Create a tqdm progress bar for the total download progress
    print(f'Downloading...')
    with tqdm(total=actual_download_size, desc='Total Download Progress', unit='B', unit_scale=True, unit_divisor=1024) as progress_bar:
        for k, file_info in enumerate(files_to_download):
            # Make sure directory exists
            os.makedirs(os.path.dirname(location + file_info['path']), exist_ok=True)

            download_url = 'https://datadryad.org' + file_info['_links']['stash:download']['href']
            with requests.get(download_url, stream=True, headers=headers) as r:
                r.raise_for_status()
                with open(location + file_info['path'], 'wb') as file:
                    for chunk in r.iter_content(chunk_size=download_chunk_size):
                        if chunk:  # filter out keep-alive new chunks
                            file.write(chunk)
                            # Update the progress bar by the size of the chunk
                            progress_bar.update(len(chunk))


    # get all file names
    chunks = [f['path'] for f in files]
    # organize alphabetically
    chunks.sort()

    # Recombine the chunks into a single file
    combined_file_name = chunks[0].split('_chunk')[0]
    with open(location + combined_file_name, 'wb') as combined_file:
        for chunk in tqdm(chunks, desc='Combining File chunks'):
            with open(location + chunk, 'rb') as file_part:
                combined_file.write(file_part.read())

    for chunk in tqdm(chunks, desc='deleting temporary files'):
        os.remove(location + chunk)

    # Extract the tar.gz file
    with tarfile.open(location + combined_file_name) as file:
    # Create a tqdm progress bar without a total
        members = []
        with tqdm(desc='Reading compressed files', unit=' files') as progress_bar:
            # Iterate over each member
            for member in file:
                members.append(member)
                # Update the progress bar for each member
                progress_bar.update(1)

    # Now extract the files
    print('Decompressing to {}...'.format(location))
    with tarfile.open(location + combined_file_name) as file:
        for member in tqdm(members, desc='Extracting Files', unit='file'):
            file.extract(member, location)

    print('Cleaning up')
    os.remove(location + combined_file_name)

    print('Complete')

    return str(Path(location).resolve()) + os.sep + combined_file_name[:-7] # name of the dataset (minus .tar.gz)

class BSCCM:

    def __init__(self, data_root, cache_index=False):
        """
        data_root: path to the top-level BSCCM directory
        cache_index: load the full index into memory. Set to true for increased performance at the expense of memory usage
        """
        if data_root[-1] != os.sep:
            data_root += os.sep
        # check if data root exists
        if not os.path.exists(data_root):
            raise ValueError('Data root {} does not exist'.format(data_root))
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
        Reads an image from the dataset.

        Args:
            index (int): The index of the image to read.
            channel (str): The name of the channel to read the image from.
            copy (bool, optional): If True, returns a copy of the image. Defaults to False.
            convert_histology_rgb32 (bool, optional): If True and the image is a histology image, 
                converts it to RGB32 format (compared to the raw histology images, which have
                3 channels each with 10-bits per pixel). Defaults to True.

        Returns:
            numpy.ndarray: The image as a numpy array.
        """
        index = int(index)
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
        """
        Returns an array of indices corresponding to the rows in the index dataframe that match the specified criteria.

        Args:
            batch (int): the batch of experiments (either 0 or 1)
            slide_replicate (int): The slide replicate number to filter by. Almost all are 0, but there are a few 1s.
            antibodies (str or list of str): The antibody or antibodies to filter by.
            has_matched_histology (bool): Whether to filter by rows that have matched histology cells.
            shuffle (bool): Whether to shuffle the resulting indices.
            seed (int): The random seed to use for shuffling. If None, the current system time is used.

        Returns:
            numpy.ndarray: An array of indices corresponding to the rows in the index dataframe that match the specified criteria.
        """
        
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
        """
        Return the shading-corrected fluorescence images for the given cell indices
        """
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
        """
        Return the surface marker data (i.e.  demixed fluorescence spectra)
        """
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
        if two_spectra_model_names[0] in self.surface_marker_dataframe.columns:
            two_spectra_data = self.surface_marker_dataframe.loc[indices][two_spectra_model_names].to_numpy()
        else:
            two_spectra_data = None
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
    
    def get_cell_type_classification_data(self, ten_class_version=False):
        """
        Get the data needed for doing cell type classification.
        This function returns the global indices of the cells, and an integer cell type label for each cell.
        By default, there are three classes: Lymphocytes (0), Granulocytes (1), and Monocytes (2).
        If the ten_class_version flag is set to True, then there are subcategories of each of these classes,
        as well as two additional classes: Red blood cells and Unknown
            Lymphocytes: 0
            Monocytes: 7 8 9
            Granulocytes: 1 2 3
            RBCs: 4
            Unknown: 5 6 
        
        These classification labels exist only for Batch 0, antibody condition "all"
        """
        marked_names = [name for name in self.surface_marker_dataframe.columns if 'selection_gated' in name]
        cluster_indices = {}
        for i, col in enumerate(marked_names):
            # Get the indices of the rows which are True for the current population
            indices = self.surface_marker_dataframe[self.surface_marker_dataframe[col]].index.tolist()
            # Store the indices in the dictionary using the population's integer key
            cluster_indices[i] = indices

        global_indices = np.concatenate(list(cluster_indices.values()))
        cluster_labels = np.concatenate([np.ones(len(indices)) * i for i, indices in cluster_indices.items()])
        if ten_class_version:
            return global_indices, cluster_labels
        else:
            # put them into the coarser categories of lymphocytes, monocytes, and granulocytes
            new_labels = []
            for label in cluster_labels:
                if label == 0:
                    new_labels.append(0)
                elif label in [1, 2, 3]:
                    new_labels.append(1)
                elif label in [7, 8, 9]:
                    new_labels.append(2)
                else:
                    new_labels.append(-1)
            new_labels = np.array(new_labels)
            mask = new_labels != -1
            return global_indices[mask], new_labels[mask]
    
    