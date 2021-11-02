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


def adjust_clc(path_clc, clc_name):
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
    clc_file = os.path.join(path_clc, clc_name)

    with rio.open(clc_file) as src:
        clc_band_1 = src.read(1)
        ras_meta = src.profile

    clc_array = np.array(clc_band_1)
    clc_array = np.where(clc_array <= 11, 100, clc_array)  # description on google drive
    clc_array = np.where(clc_array <= 22, 200, clc_array)  # description on google drive
    clc_array = np.where(clc_array <= 29, 300, clc_array)  # description on google drive
    clc_array = np.where(clc_array <= 34, 400, clc_array)  # description on google drive
    clc_array = np.where(clc_array <= 39, 500, clc_array)  # description on google drive
    clc_array = np.where(clc_array <= 47, 600, clc_array)  # description on google drive

    ras_meta.update(count=1,
                    dtype=rio.uint16)

    clc_recl_out = os.path.join(path_clc, str(clc_name[:-4] + "_reclass.tif"))
    print(clc_recl_out)
    if not os.path.isfile(clc_recl_out):
        with rio.open(path_clc + clc_name[:-4] + str("_reclass.tif"), 'w', **ras_meta) as dst:
            dst.write(clc_array, 1)


def reproject(path, filename, path_clc):
    """
    If the Sentinel-1 Data and CLC-Data have a different extent, pixel size and epsg, the function will perform a
    reprojection of CLC-data and a downsampling of the S1-Data.
    ----------
    path: string
        Path to folder with files
    file_name: string
        specific file name
    path_clc: string
        Path clc file (tif-format)
    Examples
    --------
    Returns
    -------
    raster
        convrted raster files
    """
    scene = os.path.join(path, filename)
    raster = rio.open(scene)
    print(raster.crs)

    # epsg_clc =
    # epsg_sen =

    bounds = raster.bounds
    print(bounds.left, bounds.right, bounds.top, bounds.bottom)
    # return
