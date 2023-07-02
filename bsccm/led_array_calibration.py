"""
Several helper functions for retrieving information about and plotting
the calibration of the LED array
"""
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def cartesian_to_na(point_list_cart, z_offset=8):
    """functions for calcuating the NA of an LED on the quasi-dome based on it's index for the quasi-dome illuminator
    converts a list of cartesian points to numerical aperture (NA)

    Args:
        point_list_cart: List of (x,y,z) positions relative to the sample (origin)
        z_offset : Optional, offset of LED array in z, mm

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """
    yz = np.sqrt(point_list_cart[:, 1] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)
    xz = np.sqrt(point_list_cart[:, 0] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)
    xyz = np.sqrt(point_list_cart[:, 0] ** 2 + point_list_cart[:, 1] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)

    na_xy = np.zeros((np.size(point_list_cart, 0), 2))
    na_xy[:, 0] = np.sin(np.arctan(point_list_cart[:, 0] / yz))
    na_xy[:, 1] = np.sin(np.arctan(point_list_cart[:, 1] / xz))

    xy = np.sqrt(point_list_cart[:, 0] ** 2 + point_list_cart[:, 1] ** 2 )
    na = np.sin(np.arctan(xy / (point_list_cart[:, 2] + z_offset)))
#     na = np.sin(np.arctan(np.sqrt(point_list_cart[:, 0] ** 2 
#                                   + point_list_cart[:, 1] ** 2 ) / xyz))
    
    return na_xy, na

def load_led_positions_from_json(file_name, z_offset=8):
    """Function which loads LED positions from a json file
    Args:
        fileName: Location of file to load
        zOffset : Optional, offset of LED array in z, mm
        micro : 'TE300B' or 'TE300A'
    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (x, y, z) in mm
    """
    json_data = open(file_name).read()
    data = json.loads(json_data)

    source_list_cart = np.zeros((len(data['led_list']), 3))
    x = [d['x'] for d in data['led_list']]
    y = [d['y'] for d in data['led_list']]
    z = [d['z'] for d in data['led_list']]

    source_list_cart[:, 0] = x
    source_list_cart[:, 1] = y
    source_list_cart[:, 2] = z

    source_list_naxy, na = cartesian_to_na(source_list_cart, z_offset=z_offset)

    return source_list_naxy, na, source_list_cart

def get_led_na(led_index):
    source_list_na_xy, source_list_na, source_list_cart = load_led_positions_from_json(os.path.dirname(__file__) + '/quasi_dome_design.json')
    return source_list_na[led_index - 1]
#     angles_xy = np.arcsin(np.abs(source_list_na))
#     angle = np.arctan(np.sqrt(np.tan(angles_xy[:, 0])**2 + np.tan(angles_xy[:, 1])**2 ))
#     return np.sin(angle[led_index - 1])

def get_led_na_xy(led_index):
    """
    get na x na y based on 1-based index
    :param led_index:
    :return:
    """
    source_list_na_xy, source_list_na, source_list_cart = load_led_positions_from_json(os.path.dirname(__file__) + '/quasi_dome_design.json')
#     if np.min(led_index) < 1 or np.max(led_index) > source_list_na_xy.shape[0]:
#         raise Exception('LED index out of range')
    return source_list_na_xy[led_index - 1]

def get_led_angle(led_index):
    source_list_na, source_list_cart = load_led_positions_from_json(os.path.dirname(__file__) + '/quasi_dome_design.json')
    angles_xy = np.arcsin(np.abs(source_list_na))
    angle = np.arctan(np.sqrt(np.tan(angles_xy[:, 0])**2 + np.tan(angles_xy[:, 1])**2 ))
    return angle[led_index - 1] / (2*3.14) *360

def plot_led_pattern(led_indices=None, channel_name=None, ax=None, legend=True, shorten_na_labels=False):
    """
    Make a plot of the the illumination pattern in NA space
    """
    def normalize_marker_sizes(ax, marker_size):
        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        scaling_factor = (width * height)
        return marker_size * scaling_factor
    
    if ax is None:
        ax = plt.gca()

    size = normalize_marker_sizes(ax, 2.3)

    if channel_name is None and led_indices is None:
        raise Exception('Must supply either channel_name or led_indices')
    if channel_name is not None and led_indices is not None:
        raise Exception('Must supply only channel_name or led_indices')
    if channel_name is not None:
        led_indices = illumination_to_led_indices(channel_name)

    all_led_list = np.array([get_led_na_xy(led) for led in range(1, 582)])
    ax.scatter(all_led_list[:,0], all_led_list[:,1], size, marker='o', color=[0.15,0.15,0.15], 
                facecolor=None,edgecolor=None, edgecolors=None, label='Off')
    ax.set_facecolor([0,0,0])
    
    for led_index in led_indices:
        nax, nay = get_led_na_xy(led_index) 
        ax.scatter(nax, nay, size, marker='o', color=[0,1.0,0], 
                    facecolor=None, edgecolor=None, edgecolors=None, label='On' if led_index == led_indices[0] else None)
    ax.add_patch(plt.Circle((0, 0), 0.5, alpha=0.25, label='Brightfield LEDs'))
    ax.set_xlabel('Numerical aperture (x)' if not shorten_na_labels else 'NA (x)')
    ax.set_ylabel('Numerical aperture (y)' if not shorten_na_labels else 'NA (y)')
    ax.set(xticks=[-1, 0, 1], yticks=[-1, 0, 1])
    ax.set_aspect('equal')
    if legend:
        ax.legend()


def illumination_to_led_indices(channel):
    """
    Compute a channel name to a list of LED indices. LED indices are 1-based
    """
    if channel == 'LED119':
        return [119]
    
    if 'led'in channel:
        return [int(channel.split('_')[-1])]
    
    def include_fn(na, na_xy):
        mask = np.array(na.size * [True])
        if channel == 'Brightfield':
            mask[na > 0.5] = False
        
        if 'Top' in channel:
            mask[na_xy[:, 1] < 0] = False
        elif 'Bottom' in channel:
            mask[na_xy[:, 1] > 0] = False
        elif 'Left' in channel:
            mask[na_xy[:, 0] < 0] = False
        elif 'Right' in channel:
            mask[na_xy[:, 0] > 0] = False
            
        if 'DPC' in channel:
            mask[na > 0.5] = False
            mask[na < 0.4] = False
            
        if 'DF' in channel:
            na_min = float(channel.split('_')[1]) / 100
            mask[na > na_min + 0.05] = False
            mask[na < na_min] = False
        return mask
    
    led_indices = np.arange(1, 582)
    na = get_led_na(led_indices)
    na_xy = get_led_na_xy(led_indices)
    mask = include_fn(na, na_xy)
    return led_indices[mask]
    