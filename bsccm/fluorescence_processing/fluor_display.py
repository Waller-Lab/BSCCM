from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def load_fluor_images(dataset, indices):
    f_names = ['Fluor_690-',
        'Fluor_627-673',
        'Fluor_585-625',
        'Fluor_550-570',
        'Fluor_500-550',
        'Fluor_426-446']
    # reverse order 
    f_names = f_names[::-1]
    images = []
    for i in indices:
        channels = []
        for c_index, channel in enumerate(f_names):
            channels.append(dataset.read_image(i, channel=channel))            
        images.append(np.stack(channels, axis=-1))
    
    images = np.array(images)
    return images


def make_multi_channel_fluor_images(images):
    cmaps = [LinearSegmentedColormap.from_list('testmap', [[0, 0, 0], [0, 0, 1]]),
    LinearSegmentedColormap.from_list('testmap', [[0, 0, 0], [0, 1, 0]]),
    LinearSegmentedColormap.from_list('testmap', [[0, 0, 0], [0.5, 0.5, 0]]),
    LinearSegmentedColormap.from_list('testmap', [[0, 0, 0], [1, 0, 0]]),
    LinearSegmentedColormap.from_list('testmap', [[0, 0, 0], [0.5, 0, 0.5]]),
    LinearSegmentedColormap.from_list('testmap', [[0, 0, 0], [0, 0.5, 0.5]])]
            
    ch_min = np.percentile(images, 0, axis=(0,1,2))
    ch_max = np.percentile(images, 99.95, axis=(0,1,2))


    rescaled = (images - ch_min) / (ch_max - ch_min)

    mapped = [cmaps[i](rescaled[..., i]) for i in range(6)]
    composite = np.sum(mapped, axis=0)[..., :3]
    composite[composite > 1] = 1
    return composite
