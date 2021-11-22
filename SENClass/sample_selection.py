"""
sample_selection.py: contains functions to process the geodata
@author: Felix Bachmann
"""

import pandas as pd
import numpy as np
import geodata

# so far only working for one image


def select_samples(path_clc):
    """
    The function select samples for training and testing
    ----------
    path_clc: str
        path to clc mask
    Examples
    --------
    Returns
    -------
    list
        lists containing the test and training samples for the random forrest algorithm
    """
    path = "/home/felix/PycharmProjects/SENClass/test_data/s1/S1_resamp/"
    file_name = "S1A__IW___A_20180115T182621_147_VV_grd_mli_norm_ge_resamp_100m.tif"

    file = geodata.open_raster_gdal(path, file_name)  # open s1 scene

    clc_file = geodata.open_raster_gdal(path=path_clc, file_name="CLC_subset_reclass.tif")  # open clc data

    file = np.array(file.GetRasterBand(1).ReadAsArray())  # read s1 as np.array
    mask = np.array(clc_file.GetRasterBand(1).ReadAsArray())  # read s1 as np.array

    mask_ravel = mask.ravel()  # transform 2d-array to 1d-array
    scene_ravel = file.ravel()  # transform 2d-array to 1d-array

    # print(file)

    bands = 1  # more files means more bands, as example five files/scenes means that bands equals five

    df = pd.DataFrame()  # empty df
    # now for each band (scene) all db pixel values and the corresponding clc (reclass clc) value will be collected and
    # writen to a dataframe
    for i in range(bands):
        if i == 0:
            labels = pd.Series(np.array(mask_ravel[:]).flat)  # read the class labels
            df['Label_nr'] = labels
        layer = pd.Series(np.array(scene_ravel[:]).flat)  # all pixel values of this layer
        df['Band_{}'.format(i)] = layer

    df2 = df[df.Band_0 != -99]  # remove all -99 values from data frame
    df2 = df2.dropna()  # remove all NaN values from data frame

    # now split into training and test data

    df_size = len(df2)
    test_size = int(0.1 * df_size)  # value of 0.1 gives results around 0.6
    training_size = df_size - test_size

    print(f'{training_size} pixels used for training')
    print(f'{test_size} pixels used for testing')

    lab_col = "Label_nr"
    x = df2.drop(lab_col, axis=1)
    y = df2[lab_col]
    x_test = x.iloc[0:test_size]    # get label values between first value in dataframe and threshold for testsize
    y_test = y.iloc[0:test_size]    # get backscatter values between first value in dataframe and threshold for testsize
    x_train = x.iloc[test_size+1:x.shape[0]]    # x.shape[0] max size/value from data frame
    y_train = y.iloc[test_size+1:y.shape[0]]    # x.shape[0] max size/value from data frame

    return x_train, x_test, y_train, y_test
