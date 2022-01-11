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


def sampler(nanmask, nsamples, seed):
    """
    central function to select random samples from arrays.

    Parameters
    ----------
    nanmask: numpy.ndarray
        a mask to limit the sample selection
    nsamples: int
        the number of samples to select
    seed: int
        seed used to initialize the pseudo-random number generator
    Returns
    -------
    numpy.ndarray
        the generated random samples

    See Also
    --------
    numpy.random.seed
    numpy.random.choice
    """
    indices = np.where(nanmask.flatten())[0]
    samplesize = min(indices.size, nsamples) if nsamples is not None else indices.size
    np.random.seed(seed)
    sample_ids = np.random.choice(a=indices, size=samplesize, replace=False)
    return sample_ids


def select_samples(path, path_ref_p, ref_p_name, out_folder_resampled_scenes, raster_ext, train_size, random_state,
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

    ref_p = geodata.open_raster_gdal(path_ref_p, ref_p_name)
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

    return x_train, x_test, y_train, y_test


def stratified_random_sampling(path, path_ref_p, out_folder_resampled_scenes, ref_p_name, raster_ext, random_state,
                               fraction_size, train_size, max_size, min_size):
    """

    Parameters
    ----------
    path: string
        Path to satellite images
    path_ref_p: string
        Path to reference file (mask file)
    out_folder_resampled_scenes: string
        Path to resampled satellite images
    ref_p_name: string
        Name of reference file
    raster_ext: string
        Extension from raster files
    random_state: int
        Returns a random number between 0 and 43 and ensures that the randomly selected elements are not identical in
        multiple executions.
    fraction_size: float

    train_size: float
        Size of the training split
    max_size: int
        Maximum number of samples to be drawn
    min_size: int
        Minimum number of samples to be drawn

    Returns
    -------

    """
    res_path = os.path.join(path, out_folder_resampled_scenes)
    raster_file_list, raster_file_name = geodata.parse_folder(res_path, raster_ext)
    len_ras_li = len(raster_file_list)

    ref_p = geodata.open_raster_gdal(path_ref_p, ref_p_name)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())

    file = geodata.open_raster_gdal(res_path, raster_file_name[0])
    file = np.array(file.GetRasterBand(1).ReadAsArray())

    i = 1  # 58 is faster for testing; normaly 1
    print(f'Reading {len_ras_li} raster files...')
    while i <= len_ras_li:
        test = geodata.open_raster_gdal(path, raster_file_name[i])
        test = np.array(test.GetRasterBand(1).ReadAsArray())
        file = np.append(file, test)

        ref_app = geodata.open_raster_gdal(path_ref_p, ref_p_name)
        ref_app = np.array(ref_app.GetRasterBand(1).ReadAsArray())
        ref_p = np.append(ref_p, ref_app)

        i += 1
        if i == len_ras_li:
            break
        else:
            # print(f'reading file {i}')
            continue

    print(f'Creating arrays for pixel values and label values.')

    index_val = np.where(file == -99)  # get index value from -99 values
    ref_p = np.delete(ref_p, index_val)  # delete values from array where index_val == -99
    file = np.delete(file, index_val)  # delete values from array where index_val == -99

    print(f'Creating filtered arrays \n')

    ref_p = np.asarray(ref_p)
    file = np.asarray(file)

    mask = ref_p

    mask_unique = np.unique(mask)
    mask_unique_val = np.unique(mask, return_counts=True)
    min_ax1 = np.amin(mask_unique_val, axis=1).astype(int)
    min_ax1 = min_ax1[1]
    print(f'unique values {mask_unique}')
    print(f'count of unique values {mask_unique_val[1]}')
    mask_unique_val = np.asarray(mask_unique_val[1])

    x_list = []
    y_list = []
    sampel_sum = 0
    print(f'calculating samples per class')
    for j, mask_val in enumerate(mask_unique):
        nanmask = (mask == mask_unique[j])
        nsamples = np.int(round(mask_unique_val[j] * fraction_size))
        print(f'\n Selecting samples from class {int(mask_unique[j])}')
        if nsamples > max_size:
            print(f'Using the fraction_size ({fraction_size}), {nsamples} samples would have to be drawn from the '
                  f'class. The calculated value is higher than the max_size limit. Therefore, only {max_size} samples '
                  f'are drawn for the class.')
            nsamples = max_size
        elif nsamples < max_size:
            if nsamples < min_size:
                if min_size < mask_unique_val[j]:
                    print(f'Using the fraction_size ({fraction_size}), {nsamples} samples would have to be drawn from '
                          f'the class and the class contains {mask_unique_val[j]} values. Since the calculated value is'
                          f' less than the min_size value ({min_size}), {min_size} samples are drawn from the class')
                    nsamples = min_size
                else:
                    print(f'The calculated number of samples to be drawn ({nsamples}) is less than the min_size value '
                          f'{min_size}. 90% of the available pixels in the class are used as samples. This will '
                          f'draw {np.int(round(mask_unique_val[j] * 0.9))} samples from {mask_unique_val[j]} values '
                          f'in the class')
                    nsamples = np.int(round(mask_unique_val[j] * 0.9))
            else:
                print(f'{nsamples} were calculated for the class and are drawn.')
                nsamples = nsamples

        sampel_sum = sampel_sum + nsamples

        seed = random_state
        sample_ids = sampler(nanmask, nsamples, seed)

        x = file.flatten()[sample_ids]
        y = mask.flatten()[sample_ids]

        x_list.append(x)
        y_list.append(y)

    print(f'\n In total {sampel_sum} samples were selected.')

    flat_list_x = []
    for sublist in x_list:
        for item in sublist:
            flat_list_x.append(item)

    flat_list_y = []
    for sublist in y_list:
        for item in sublist:
            flat_list_y.append(item)

    df = pd.DataFrame(list(zip(flat_list_x, flat_list_y)),
                      columns=['x_val', 'y_val'])

    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    x = x.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state)

    return x_train, x_test, y_train, y_test


def random_sampling(path, path_ref_p, out_folder_resampled_scenes, ref_p_name, raster_ext,
                    random_state, train_size, n_samples):
    """

    Parameters
    ----------
    path: string
        Path to satellite images
    path_ref_p: string
        Path to reference file (mask file)
    out_folder_resampled_scenes: string
        Path to resampled satellite images
    ref_p_name: string
        Name of reference file
    raster_ext: string
        Extension from raster files
    random_state: int
        Returns a random number between 0 and 43 and ensures that the randomly selected elements are not identical in
        multiple executions.
    train_size: float
        Size of the training split
    n_samples: int
        Number of samples to be drawn per class

    Returns
    -------

    """
    res_path = os.path.join(path, out_folder_resampled_scenes)
    raster_file_list, raster_file_name = geodata.parse_folder(res_path, raster_ext)
    len_ras_li = len(raster_file_list)

    ref_p = geodata.open_raster_gdal(path_ref_p, ref_p_name)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())

    file = geodata.open_raster_gdal(res_path, raster_file_name[0])
    file = np.array(file.GetRasterBand(1).ReadAsArray())

    i = 55 # 58 is faster for testing; normaly 1
    print(f'Reading {len_ras_li} raster files...')
    while i <= len_ras_li:
        test = geodata.open_raster_gdal(res_path, raster_file_name[i])
        test = np.array(test.GetRasterBand(1).ReadAsArray())
        file = np.append(file, test)

        ref_app = geodata.open_raster_gdal(path_ref_p, ref_p_name)
        ref_app = np.array(ref_app.GetRasterBand(1).ReadAsArray())
        ref_p = np.append(ref_p, ref_app)

        i += 1
        if i == len_ras_li:
            break
        else:
            # print(f'reading file {i}')
            continue

    print(f'Creating arrays for pixel values and label values.')

    index_val = np.where(file == -99)  # get index value from -99 values
    ref_p = np.delete(ref_p, index_val)  # delete values from array where index_val == -99
    file = np.delete(file, index_val)  # delete values from array where index_val == -99

    print(f'Creating filtered arrays \n')

    mask = ref_p

    mask_unique = np.unique(mask)
    mask_unique_val = np.unique(mask, return_counts=True)
    print(f'unique values in reference product {mask_unique}')
    print(f'count of unique values in reference product {mask_unique_val[1]} \n')
    mask_unique_val = np.asarray(mask_unique_val[1])

    x_list = []
    y_list = []
    sampel_sum = 0
    for j, mask_val in enumerate(mask_unique):
        nanmask = (mask == mask_unique[j])

        if n_samples > mask_unique_val[j]:
            nsamples = np.int(round(mask_unique_val[j] * 0.9))
            print(
                f'The specified sample size {n_samples} is chosen too large, because in class {int(mask_unique[j])} \n'
                f'only {mask_unique_val[j]} values are available and is smaller than the specified sample size \n'
                f'({n_samples}). 90% of the available pixels in the class are used as samples. This will draw \n'
                f'{np.int(round(mask_unique_val[j] * 0.9))} samples from {mask_unique_val[j]} values in the class. \n')
        else:
            nsamples = n_samples

        sampel_sum = sampel_sum + nsamples

        seed = random_state
        sample_ids = sampler(nanmask, nsamples, seed)

        x = file.flatten()[sample_ids]
        y = mask.flatten()[sample_ids]

        x_list.append(x)
        y_list.append(y)

    x_list = np.asarray(x_list).flatten()
    y_list = np.asarray(y_list).flatten()

    flat_list_x = []
    for sublist in x_list:
        for item in sublist:
            flat_list_x.append(item)

    flat_list_y = []
    for sublist in y_list:
        for item in sublist:
            flat_list_y.append(item)

    df = pd.DataFrame(list(zip(flat_list_x, flat_list_y)),
                      columns=['x_val', 'y_val'])

    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    x = x.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state)

    return x_train, x_test, y_train, y_test
