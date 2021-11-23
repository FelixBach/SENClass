"""
geodata.py: contains functions to process the geodata
@author: Felix Bachmann
"""

import rasterio as rio
import os
import numpy as np
import glob
import gdal
import osr


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
        raster_file_name = [w[len(path):-(len(raster_ext) + 1)] for w in raster_file_list]

    # print(raster_file_list)
    print(f'folder contains {len(raster_file_list)} raster files \n')

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


def reclass_clc(path_ref_p, clc_name):
    """
    The CLC values are divided into five new classes.

    Parameters
    ----------
    path_ref_p: string
        Path clc file (tif-format)
    clc_name: string
        Specific file name
    Examples
    --------
    Returns
    -------
    """
    clc_file = os.path.join(path_ref_p, clc_name)

    with rio.open(clc_file) as src:
        clc_band_1 = src.read(1)
        ras_meta = src.profile

    clc_array = np.array(clc_band_1)
    clc_array = np.where(clc_array <= 11, 100, clc_array)  # description on google drive
    clc_array = np.where(clc_array <= 22, 200, clc_array)  # description on google drive
    # clc_array = np.where(clc_array <= 29, 300, clc_array)  # description on google drive
    clc_array = np.where(clc_array <= 34, 400, clc_array)  # description on google drive
    clc_array = np.where(clc_array <= 39, 500, clc_array)  # description on google drive
    clc_array = np.where(clc_array <= 47, 600, clc_array)  # description on google drive

    ras_meta.update(count=1,
                    dtype=rio.uint16)

    clc_recl_out = os.path.join(path_ref_p, str(clc_name[:-4] + "_reclass.tif"))
    if not os.path.isfile(clc_recl_out):
        with rio.open(path_ref_p + clc_name[:-4] + str("_reclass.tif"), 'w', **ras_meta) as dst:
            dst.write(clc_array, 1)

    print(f'reclassified clc data output file: {clc_recl_out} \n')
    return clc_recl_out


def reproject(path, ref_p_reclass, raster_file_list, raster_file_name):
    """
    If the Sentinel-1 Data and CLC-Data have a different extent, pixel size and epsg, the function will perform a
    reprojection of CLC-data and a downsampling of the S1-Data.
    The CLC file is processed individually. Since the geometric resolution is 100m, only the coordinate system is
    adjusted. For this the coordinate system is read from the first scene in the raster_file_list and then taken over
    at gdal.Wrap. Afterwards all sentinel scenes are adjusted. Since the CLC file got the coordinate system of the
    sentinel scenes, the geometric resolution is adapted to that of the CLC data. For this purpose, the pixel size is
    read from the CLC data and inserted into gdal.wrap accordingly.

    Parameters
    ----------
    path: string
        Path to folder with files
    ref_p_reclass: string
        Path to the clc file (tif-format)
    raster_file_list: list
        list with paths to Sentinel scenes
    raster_file_name: list
        name from each file in the raster_file_list
    Examples
    --------
    Returns
    -------
    """
    clc_file = ref_p_reclass
    clc = gdal.Open(clc_file)
    s1 = gdal.Open(raster_file_list[0])

    proj_clc = osr.SpatialReference(wkt=clc.GetProjection())
    epsg_clc = proj_clc.GetAttrValue('AUTHORITY', 1)
    gt_clc = clc.GetGeoTransform()
    pix_size_clc = gt_clc[1]

    proj_s1 = osr.SpatialReference(wkt=s1.GetProjection())
    epsg_s1 = proj_s1.GetAttrValue('AUTHORITY', 1)
    gt_s1 = s1.GetGeoTransform()
    pix_size_s1 = gt_s1[1]

    gt = s1.GetGeoTransform()
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * s1.RasterXSize
    miny = maxy + gt[5] * s1.RasterYSize

    clc_res = gdal.Warp('', clc, format='VRT', dstSRS='EPSG:{}'.format(epsg_s1), xRes=pix_size_clc, yRes=pix_size_clc,
                        outputType=gdal.GDT_Int16, outputBounds=[minx, miny, maxx, maxy])

    out_clc = clc_file[:-4] + str("_reprojected.tif")
    write_file_gdal(clc_res, out_clc)
    print(f'reprojected clc file from EPSG {epsg_clc} to EPSG {epsg_s1} \n'
          f'output file: {out_clc}\n')

    for i, raster in enumerate(raster_file_list):
        s1 = gdal.Open(raster_file_list[i])

        gt_clc = clc.GetGeoTransform()
        psize_clc = gt_clc[1]

        gt = s1.GetGeoTransform()
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * s1.RasterXSize
        miny = maxy + gt[5] * s1.RasterYSize

        s1_res = gdal.Warp('', s1, format='VRT', xRes=psize_clc, yRes=psize_clc,
                           outputType=gdal.GDT_Float32, outputBounds=[minx, miny, maxx, maxy])

        out_folder = "S1_resamp"
        out_folder = os.path.join(path, out_folder)

        if not os.path.isdir(out_folder):   # create directory for resampled Sentinel scenes
            os.makedirs(out_folder)

        file_name = raster_file_name[i][:-4] + str("_resamp_100m.tif")
        out_file = os.path.join(out_folder, file_name)   # out_folder + file_name

        write_file_gdal(s1_res, out_file)

        print(f'resampled {i+1} scenes from {pix_size_s1}m to {psize_clc}m')


def array_to_gtiff(pred):
    """
    The function writes a numpy array to a GTIFF tile
    Parameters
    ----------
    Examples
    --------
    Returns
    -------
    """
    driver = gdal.GetDriverByName("GTIFF")
    rows = pred.shape
    print(rows)

    return
