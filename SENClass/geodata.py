"""
geodata.py: contains functions to process the geodata
@author: Felix Bachmann
"""

import os
import osr
import glob
import gdal
import numpy as np


def parse_folder(path, raster_ext):
    """
    Returns a list with all raster files in a folder
    Parameters
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
    raster_file_name = []
    for file in glob.glob(path + "*" + raster_ext):
        raster_file_list.append(file)
        raster_file_list = [w.replace('\\', '/') for w in raster_file_list]
        raster_file_name = [w[len(path):1000] for w in raster_file_list]

    return raster_file_list, raster_file_name


def open_raster_gdal(path, file_name):
    """
    The function opens raster files from folder/raster_file_list
    Parameters
    ----------
    path: string
        Path to folder with raster files
    file_name:
        name of a raster file or name from a list
    Examples
    --------
    Returns
    -------
    Gdal file object
    """

    file_name = os.path.join(path, file_name)
    gdal_file = gdal.Open(file_name)

    return gdal_file


def write_file_gdal(gdal_file, out_file):
    """
    Saves gdal files to disk.
    Parameters
    ----------
    gdal_file: GDAL file object
        File object containing the data to be written to disk
    out_file: str
        Path and raster name for the new file
    Examples
    --------
    Returns
    -------
    """
    driver = gdal.GetDriverByName('GTIFF')
    cols = gdal_file.RasterYSize
    rows = gdal_file.RasterXSize
    bands = gdal_file.RasterCount

    dtype = gdal.GDT_Float32
    out_file = driver.Create(out_file, rows, cols, bands, dtype)
    out_file.SetGeoTransform(gdal_file.GetGeoTransform())
    out_file.SetProjection(gdal_file.GetProjection())

    for i in range(bands):
        band = gdal_file.GetRasterBand(i + 1).ReadAsArray()
        out_file.GetRasterBand(i + 1).WriteArray(band)

    out_file.FlushCache()  # save from memory to disk

    return


def reclass_raster(out_ref_p):
    """
        The reference product values are divided into new classes.
    Parameters
    ----------
    out_ref_p: string
        path to resampled/reclassified reference product
    Returns
    -------
    """
    ref_p = gdal.Open(out_ref_p)

    gt = ref_p.GetGeoTransform()
    prj = ref_p.GetProjection()
    srs = osr.SpatialReference(wkt=prj)

    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())
    ref_p = np.where(ref_p <= 0, 100, ref_p)
    ref_p = np.where(ref_p <= 11, 200, ref_p)  # description on google drive
    ref_p = np.where(ref_p <= 12, 300, ref_p)  # description on google drive

    driver = gdal.GetDriverByName('GTIFF')
    rows, cols = ref_p.shape
    out_ds = driver.Create(out_ref_p, cols, rows, 1, gdal.GDT_UInt16)
    out_ds.GetRasterBand(1).WriteArray(ref_p)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(srs.ExportToWkt())
    out_ds = None

    print(f'reclassified reference product')


def reproject_raster(path, path_ref_p, ref_p_name, raster_ext, out_folder_resampled_scenes):
    """
    The raster used as reference product is projected into the coordinate system of the satellite images. The satellite
    images are not reprojected, but the pixel size is adjusted to that of the reference product.
    ----------
    path: string
        Path to folder with satellite files
    path_ref_p: string
        Path to the ref_p file (tif-format)
    ref_p_name: string
        list with paths to satellite files
    raster_ext: string
        extension from raster files
    out_folder_resampled_scenes: string
        path for the output folder with the resampled scenes
    Returns
    -------
    """
    # search for files in input folder
    raster_file_list, raster_file_name = parse_folder(path, raster_ext)
    print(f'folder contains {len(raster_file_list)} raster files \n')

    # get spatial reference from reference product
    ref_p_file = os.path.join(path_ref_p, ref_p_name)
    ref_p = gdal.Open(ref_p_file)
    proj_ref_p = osr.SpatialReference(wkt=ref_p.GetProjection())
    epsg_ref_p = proj_ref_p.GetAttrValue('AUTHORITY', 1)

    # get spatial reference from satellite data
    s1 = gdal.Open(raster_file_list[0])
    proj_s1 = osr.SpatialReference(wkt=s1.GetProjection())
    epsg_s1 = proj_s1.GetAttrValue('AUTHORITY', 1)
    gt_s1 = s1.GetGeoTransform()
    pix_size_s1 = gt_s1[1]

    gt = s1.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * s1.RasterXSize
    miny = maxy + gt[5] * s1.RasterYSize

    ref_p_res = gdal.Warp('', ref_p, format='VRT', dstSRS='EPSG:{}'.format(epsg_s1), outputType=gdal.GDT_Int16,
                          outputBounds=[minx, miny, maxx, maxy])

    # writing reprojected reference product to disk
    out_ref_p = ref_p_file[:-4] + str("_reprojected.tif")
    write_file_gdal(ref_p_res, out_ref_p)
    print(f'reprojected ref_p file from EPSG {epsg_ref_p} to EPSG {epsg_s1} \n'
          f'output file: {out_ref_p}\n')

    # resampling satellite data
    ref_p = gdal.Open(out_ref_p)
    for i, raster in enumerate(raster_file_list):
        s1 = gdal.Open(raster_file_list[i])

        gt_ref_p = ref_p.GetGeoTransform()
        psize_ref_p = gt_ref_p[1]

        gt = s1.GetGeoTransform()
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * s1.RasterXSize
        miny = maxy + gt[5] * s1.RasterYSize

        s1_res = gdal.Warp('', s1, format='VRT', xRes=psize_ref_p, yRes=psize_ref_p,
                           outputType=gdal.GDT_Float32, outputBounds=[minx, miny, maxx, maxy])

        out_folder = out_folder_resampled_scenes
        out_folder = os.path.join(path, out_folder)

        # create directory for resampled Sentinel scenes
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        file_name = raster_file_name[i] + str("_resampled.tif")
        file_check = os.path.join(out_folder, file_name)
        if not os.path.isfile(file_check):
            out_file = os.path.join(out_folder, file_name)
            write_file_gdal(s1_res, out_file)
            print(f'resampled {i + 1} scenes from {pix_size_s1}m to {round(psize_ref_p, 2)}m')
        else:
            print(f'resampled file already exists')

    return out_ref_p


def prediction_to_gtiff(prediction, path, out_folder_prediction, name_predicted_image, out_ref_p, raster_ext, mask):
    """
        The function writes the predicted array to a GTIFF file.
    Parameters
    ----------
    prediction
    path
    out_folder_prediction
    name_predicted_image
    out_ref_p
    raster_ext
    Returns
    -------
    """
    # creates output file
    out_folder = os.path.join(path, out_folder_prediction)

    # create directory for resampled Sentinel scenes
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    file_name = out_folder + str("name_predicted_image") + str(".") + raster_ext
    file_check = os.path.join(out_folder, file_name)
    if not os.path.isfile(file_check):
        # read meta information from reference product
        ref_p = gdal.Open(out_ref_p)
        gt = ref_p.GetGeoTransform()
        prj = ref_p.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())
        cols, rows = ref_p.shape

        # reshaping prediction and saving raster with meta information from reference product to disk
        grid = prediction.reshape((cols, rows))
        driver = gdal.GetDriverByName('GTIFF')
        rows, cols = ref_p.shape
        grid = grid.astype('float32')

        #apply an edge mask with NaN values
        grid[mask] = np.nan

        out_ds = driver.Create(file_name, cols, rows, 1, gdal.GDT_UInt16)
        # writing output raster
        out_ds.GetRasterBand(1).WriteArray(grid)
        out_ds.SetGeoTransform(gt)
        # setting spatial reference of output raster
        out_ds.SetProjection(srs.ExportToWkt())
        # Close output raster dataset
        out_ds = None
        print(f'GTIFF created from predicted labels')
    else:
        print(f'predicted image already exists')
    return