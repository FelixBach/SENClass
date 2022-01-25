"""
sample_selection.py: contains functions select training and test samples
@author: Felix Bachmann
"""

import os
import gdal
import geodata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def select_samples(path, path_ref_p, out_ref_p, out_folder_resampled_scenes, raster_ext, train_size, random_state,
                   strat):
    """
    The function select samples for training and testing. The user has the choice between two methods to select the
    test and training pixels, as well as test and training labels from the data frame. If strat is set to true, the
    pixels and labels are selected using stratified random sampling.  Otherwise, the pixels and labels are randomly
    selected from the data frame.
    ----------
    path_ref_p: string
        path to reference file (mask file)
    ref_p_name: string
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
    strat: bool
        If True, then stratified random sampling is performed. If strat is set to False, then a random sampling is
        performed.
    Examples
    --------
    Returns
    -------
    list
        lists containing the test and training samples for the random forest algorithm.
    """
    res_path = os.path.join(path, out_folder_resampled_scenes)
    raster_file_list, raster_file_name = geodata.parse_folder(res_path, raster_ext)
    print(f'folder contains {len(raster_file_list)} resampled raster files \n')

    len_ras_li = len(raster_file_list)  # number of satellite images

    ref_p = geodata.open_raster_gdal(path_ref_p, out_ref_p)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())

    print(f"Creating data frame with labels and pixel values from satellite images")

    df = pd.DataFrame()
    labels = pd.Series(np.array(ref_p[:]).flat)  # read the class labels
    df['Label_nr'] = labels

    for i in range(len_ras_li):
        file = gdal.Open(raster_file_list[i])
        file = np.array(file.GetRasterBand(1).ReadAsArray())
        file = file.flatten()
        layer = pd.Series(np.array(file).flat)
        df['file_{}'.format(i)] = layer

    data = df
    # print(df.shape)
    df2 = df[df != -99]  # remove all -99 values from data frame
    df2 = df2.dropna()  # remove all NaN values from data frame
    # print(df2.shape)

    print(f"Removing -99 and NaN-values from data frame")

    if strat:
        print(f"Using stratified random sample selection")
        print(f"creating seperate data frames for labels and pixel values \n")
        df_y = df2.pop("Label_nr")
        df_x = df2

        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, train_size=train_size, random_state=random_state
                                                            , stratify=df_y)
        print(f"{len(x_train)} pixels used for training and {len(x_test)} pixels used for testing \n")

    else:
        print(f"Using random sample selection")
        row_count = df2.shape[1]  # get max rows from data frame

        x = df2.iloc[:, 1:row_count].values
        y = df2.iloc[:, 0].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state)

        print(f"{len(x_train)} pixels used for training and {len(x_test)} pixels used for testing \n")

    return x_train, x_test, y_train, y_test, data
