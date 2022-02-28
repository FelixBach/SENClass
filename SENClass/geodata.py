"""
geodata.py: contains functions to process geodata
@author: Felix Bachmann, Anastasiia Vynohradova
"""

import os
import osr
import glob
import gdal
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def parse_folder(path, raster_ext):
    """
    Returns a list with all raster files in a folder

    Parameters
    ----------
    path: string
        Path to folder with raster files
    raster_ext: string
        specifies raster format

    Returns
    -------
    raster_file_list: list
        list of all raster files in a folder
    raster_file_name: list
        list of all raster names in a folder
    """
    print('\n##########   -   Searching for files   -   ##########')
    raster_file_list = []
    raster_file_name = []

    # parsing folder for files
    for file in glob.glob(path + "*" + raster_ext):
        raster_file_list.append(file)
        raster_file_list = [w.replace('\\', '/') for w in raster_file_list]
        raster_file_name = [w[len(path):1000] for w in raster_file_list]

    return raster_file_list, raster_file_name


def open_raster_gdal(path, file_name):
    """
    The function opens raster files from folder or raster_file_list

    Parameters
    ----------
    path: string
        Path to folder with raster files
    file_name: string
        name of a raster file or file from a raster_file_list

    Returns
    -------
    gdal_file: Gdal file object
        Raster file that can be read as array with numpy
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
    out_file: string
        Path and raster name for the new file

    Returns
    -------
    This function has no return
    """
    # call driver
    driver = gdal.GetDriverByName('GTIFF')

    # define pixel size and number of badns
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


def reclass_raster(raster_value, class_value, out_ref_p):
    """
    The reference product values are divided into new classes. For that a simple reclassification approach is used. To
    assign new classes, the smallest value in list class_value has to bei higher than the highest value in the
    raster_value list. Otherwise, the reclassification will be wrong. It's also important that both lists have the same
    length. As user, you have to know the class values of your product.

    Parameters
    ----------
    raster_value: list
        list that contains the raster values as class limits
    class_value: list
        list with new class values
    out_ref_p: string
        path to resampled/reclassified reference product

    Returns
    -------
    This function has no return
    """
    print('\n##########   -   Reclassifying data   -   ##########')

    # check if lists are of the same length
    if len(class_value) == len(raster_value):
        ref_p = gdal.Open(out_ref_p)

        gt = ref_p.GetGeoTransform()
        prj = ref_p.GetProjection()
        srs = osr.SpatialReference(wkt=prj)

        # reclassifying with loop
        ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())
        for i, value in enumerate(raster_value):
            ref_p = np.where(ref_p <= raster_value[i], class_value[i], ref_p)

        # saving file
        driver = gdal.GetDriverByName('GTIFF')
        rows, cols = ref_p.shape
        out_ds = driver.Create(out_ref_p, cols, rows, 1, gdal.GDT_UInt16)
        out_ds.GetRasterBand(1).WriteArray(ref_p)
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(srs.ExportToWkt())
        out_ds = None
        print(f'reclassified reference product')
    else:
        print(f'list raster_value and class_value do not have the same length. No reclassification possible. Continue '
              f'with sample selection.')
        pass


def reproject_raster(path, path_ref_p, ref_p_name, raster_ext, out_folder_resampled_scenes):
    """
    The raster used as reference product is projected into the coordinate system of the satellite images. The satellite
    images are not reprojected, but the pixel size is adjusted to that of the reference product.

    Parameters
    ----------
    path: string
        Path to folder with raster files
    path_ref_p: string
        Path to the ref_p file (tif-format)
    ref_p_name: string
        list with paths to satellite files
    raster_ext: string
        specifies raster format
    out_folder_resampled_scenes: string
        path for the output folder with the resampled scenes

    Returns
    -------
    This function has no return
    """
    print('\n####################   -   Preparing the Geodata   -   ####################')
    # search for files in input folder
    raster_file_list, raster_file_name = parse_folder(path, raster_ext)
    print(f'{path} contains {len(raster_file_list)} raster files \n')

    print('\n##########   -   Reprojecting data   -   ##########')
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

    minx = gt_s1[0]
    maxy = gt_s1[3]
    maxx = minx + gt_s1[1] * s1.RasterXSize
    miny = maxy + gt_s1[5] * s1.RasterYSize

    # performing reprojecting
    ref_p_res = gdal.Warp('', ref_p, format='VRT', dstSRS='EPSG:{}'.format(epsg_s1), outputType=gdal.GDT_Int16,
                          outputBounds=[minx, miny, maxx, maxy])

    # writing reprojected reference product to disk
    out_ref_p = ref_p_file[:-4] + str("_reprojected.tif")
    write_file_gdal(ref_p_res, out_ref_p)
    print(f'reprojected {ref_p_file} file from EPSG {epsg_ref_p} to EPSG {epsg_s1} \n'
          f'location from resampled file: {out_ref_p}\n')

    # create directory for resampled Sentinel scenes
    out_folder = out_folder_resampled_scenes
    out_folder = os.path.join(path, out_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

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

        # performing reprojecting for each scene
        s1_res = gdal.Warp('', s1, format='VRT', xRes=psize_ref_p, yRes=psize_ref_p,
                           outputType=gdal.GDT_Float32, outputBounds=[minx, miny, maxx, maxy])

        # check for file
        file_name = raster_file_name[i][:-4] + str("_resampled.tif")
        file_check = os.path.join(out_folder, file_name)
        if not os.path.isfile(file_check):
            out_file = os.path.join(out_folder, file_name)
            write_file_gdal(s1_res, out_file)
            print(f'resampled {i + 1} scenes from {pix_size_s1}m to {round(psize_ref_p, 2)}m')
        else:
            print(f'resampled file already exists')

    print(f'location from resampled satellite images: {out_folder}\n')
    return out_ref_p


def select_samples(path, path_ref_p, out_ref_p, out_folder_resampled_scenes, raster_ext, train_size, random_state,
                   sss):
    """
    The function select samples for training and testing. The user has the choice between two methods to select the
    test and training pixels. If strat is set to true, the pixels and labels are selected using the sklearn algorithm
    StratifiedShuffleSplit. Otherwise, the pixels and labels are randomly selected from the data frame using the
    sklearn algorithm train_test_split.

    Parameters
    ----------
    path_ref_p: string
        path to reference file (mask file)
    out_ref_p: string
        name of reference file
    path: string
        path to satellite images
    out_folder_resampled_scenes: string
        path to resampled satellite images
    raster_ext: string
        extension from raster files
    train_size: string
        size of the test
    random_state: int
        Returns a random number between 0 and 43 and ensures that the randomly selected elements are not identical in
        multiple executions.
    sss: bool
        If True, sk-learn function StratifiedRandomSampling is used to select the samples. If sss is false, sk-learn
        uses train_test_split to select the samples.

    Returns
    -------
    x_train: list
        list containing the training samples from the satellite images for the random forest algorithm.
    y_train: list
        list containing the training samples from the reference product for the random forest algorithm.
    data: pandas.core.frame.DataFrame
        data on which the prediction is executed
    mask: numpy.ndarray
        array to mask -99 values
    """
    print('\n####################   -   Start sample selection   -   ####################')
    global x_train, y_train

    # search for reprojected files
    res_path = os.path.join(path, out_folder_resampled_scenes)
    raster_file_list, raster_file_name = parse_folder(res_path, raster_ext)
    print(f'{res_path} contains {len(raster_file_list)} resampled raster files \n')

    print('\n##########   -   Preparing data   -   ##########')

    len_ras_li = len(raster_file_list)  # number of satellite images

    ref_p = open_raster_gdal(path_ref_p, out_ref_p)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())

    print(f"Creating data frame with labels and pixel values from satellite images")
    # saving reference product to DataFrame
    df = pd.DataFrame()
    labels = pd.Series(np.array(ref_p[:]).flat)  # read the class labels
    df['Label_nr'] = labels

    # saving satellite images to DataFrame
    for i in range(len_ras_li):
        file = gdal.Open(raster_file_list[i])
        file = np.array(file.GetRasterBand(1).ReadAsArray())
        file = file.flatten()
        layer = pd.Series(np.array(file).flat)
        df['file_{}'.format(i)] = layer

    # create an edge mask with NaN values
    raster = gdal.Open(raster_file_list[1])
    raster = np.array(raster.GetRasterBand(1).ReadAsArray())
    raster[raster == -99] = np.nan
    mask = np.isnan(raster)

    # cleaning data frame
    data = df
    data = data.iloc[:, 1:]
    df2 = df[df != -99]  # remove all -99 values from data frame
    df2 = df2.dropna()  # remove all NaN values from data frame

    print(f"Removing -99 and NaN-values from data frame")

    # selecting sampels from DataFrame
    if sss:
        print('\n##########   -   Using StratifiedShuffleSplit from sklearn.model_selection  -   ##########')
        row_count = df2.shape[1]  # get max rows from data frame
        x = df2.iloc[:, 1:row_count].values
        y = df2.iloc[:, 0].values
        sss = StratifiedShuffleSplit(n_splits=10, train_size=train_size, random_state=random_state)
        for train_index, test_index in sss.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

        print(f"{len(x_train)} pixels used for training \n")
        return x_train, y_train, data, mask

    else:
        print('\n##########   -   Using train_test_split from sklearn.model_selection   -   ##########')
        row_count = df2.shape[1]  # get max rows from data frame

        x = df2.iloc[:, 1:row_count].values
        y = df2.iloc[:, 0].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state,
                                                            stratify=y)

        print(f"{len(x_train)} pixels used for training \n")

        return x_train, y_train, data, mask


def prediction_to_gtiff(prediction, path, out_folder_prediction, name_predicted_image, out_ref_p, raster_ext, mask):
    """
    The function writes the predicted array to a GTIFF file.

    Parameters
    ----------
    prediction: numpy.ndarray
        Numpy-Array with the predicted labels
    path: string
        Path to folder with raster files
    out_folder_prediction: string
        Name for the output folder for the predicted image
    name_predicted_image: string
        Name of the predicted image
    out_ref_p: string
        path to resampled/reclassified reference product
    raster_ext: string
        specifies raster format
    mask: numpy.ndarray
        array to mask -99 values

    Returns
    -------
    This function has no return
    """
    # create name for output folder
    out_folder = os.path.join(path, out_folder_prediction)

    # create directory for predicted image
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # create file name and check for existing
    file_name = out_folder + name_predicted_image + str(".") + raster_ext
    file_check = os.path.join(out_folder, file_name)
    if not os.path.isfile(file_check):
        print('\n##########   -   save predicted image   -   ##########')
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

        # apply an edge mask with NaN values
        grid[mask] = np.nan

        nodata = 0
        out_ds = driver.Create(file_name, cols, rows, 1, gdal.GDT_UInt16)
        # writing output raster
        out_ds.GetRasterBand(1).WriteArray(grid)
        out_ds.GetRasterBand(1).SetNoDataValue(nodata)
        out_ds.SetGeoTransform(gt)
        # setting spatial reference of output raster
        out_ds.SetProjection(srs.ExportToWkt())
        # Close output raster dataset
        out_ds = None
        print(f'GTIFF created from predicted labels')
    else:
        print(f'predicted image already exists')


def tif_visualize(path, out_folder_prediction, filename, raster_ext):
    """
    Visualizes the result as image. The function only visualizes thr result from the water seasonality product.

    Parameters
    ----------
    path: string
        Path to folder with raster files
    out_folder_prediction: string
       Path to the folder where the prediction should be saved
    filename: string
        Name from the file that is stored
    raster_ext: string
        specifies raster format

    Returns
    -------
    This function has no return
    """
    # open result
    result = rasterio.open(os.path.join(path, out_folder_prediction, filename + "." + raster_ext))

    # creating plot and legend
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = mpl.colors.ListedColormap(['white', 'cyan', 'royalblue'])
    white_box = mpatches.Patch(color='white', label='not flooded')
    blue_box = mpatches.Patch(color='cyan', label='temporarily flooded')
    red_box = mpatches.Patch(color='royalblue', label='permanently flooded')
    ax.legend(handles=[white_box, blue_box, red_box], title='Classes', handlelength=0.7, bbox_to_anchor=(1.05, 0.45),
              loc='lower left', borderaxespad=0.)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # plot result
    show(result,
         transform=result.transform, title="Classification result", ax=ax, cmap=cmap)
    plt.show()
