"""
geodata.py: contains functions to process the geodata
@author: Felix Bachmann
"""

import os
import rasterio as rio
import numpy as np
import glob


def parse_folder(path, raster_ext):
    """
    Returns a list with all raster files in a folder
    ----------
    path: string
        Path to folder with raster files
    raster_ext: string
        specifies raster format

    Examples
    --------
    Returns
    -------
    list
        list of all raster files in folder
    """
    raster_file_list = []
    for file in glob.glob(path + "*" + raster_ext):
        raster_file_list.append(file)
        raster_file_list = [w.replace('\\', '/') for w in raster_file_list]
        raster_file_name = [w[len(path):-(len(raster_ext) + 1)] for w in raster_file_list]

    return print(len(raster_file_list)), raster_file_list


def open_raster(path, file_name):  # later instead of path and file_name raster_file_list
    """
    The function opens raster files from folder/raster_file_list
    ----------
    path: string
        Path to folder with raster files
    raster_ext: string
        specifies raster format

    Examples
    --------
    Returns
    -------
    """
    file = os.path.join(path, file_name)
    with rio.open(file) as src:
        band_1 = src.read(1)

    # print(f'max band_1 {np.nanmax(band_1)}') # just for testing


def adjust_clc(path_clc):
    """
    The CLC values are divided into six new classes
    ----------
    path_clc: string
        Path clc file (tif-format)
    Examples
    --------
    Returns
    -------
    """
    with rio.open(path_clc) as src:
        clc_band_1 = src.read(1)

    clc_array = np.array(clc_band_1)
    clc_array = np.where(clc_array <= 11, 100, clc_array)
    clc_array = np.where(clc_array <= 22, 200, clc_array)
    clc_array = np.where(clc_array <= 29, 300, clc_array)
    clc_array = np.where(clc_array <= 34, 400, clc_array)
    clc_array = np.where(clc_array <= 39, 500, clc_array)
    clc_array = np.where(clc_array <= 47, 600, clc_array)

    print(clc_band_1)
    print(clc_array)
