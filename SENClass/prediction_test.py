import os
import gdal
import geodata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble as ensemble
import matplotlib.pyplot as plt
import gdal
import osr
from scipy.spatial import distance as dist
import seaborn as sb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def calcClassHist(data, label_col="Label"):
    """
    calculate histograms per class
    Parameters
    ----------
    data: numpy.dataFrame
        dataframe containing dataset
    label_col: str (optional)
        name of label column
    Examples
    --------
    Returns
    -------
    accuracy score of prediction
    """
    histograms = {}
    classes = data[label_col].unique()
    for class_name in classes:
        class_vals = data.loc[data[label_col] == class_name]
        class_vals = class_vals.drop(label_col, 1)
        hist, bins = np.histogram(class_vals)
        histograms[class_name] = hist
    return histograms


def calcSep(data, label_col="Label"):
    """
    calculate class seperability with different measures
    (Euclidean, Manhatten, Chebyshev)
    Parameters
    ----------
    data: numpy.dataFrame
        dataframe containing dataset
    label_col: str (optional)
        name of label column
    Examples
    --------
    Returns
    -------
    dict
        dictionary of arrays containing distance measures for all classes
    """
    comparison = {}
    SCIPY_METHODS = {
        "Euclidean": dist.euclidean,
        "Manhattan": dist.cityblock,
        "Chebyshev": dist.chebyshev
    }
    histograms = calcClassHist(data, label_col)
    for (methodName, method) in SCIPY_METHODS.items():
        results = {}
        for (class_name, hist) in histograms.items():
            class_results = []
            for (n, nhist) in histograms.items():
                d = method(histograms[class_name], nhist)
                class_results.append(d)
            results[class_name] = class_results
        comparison[methodName] = results
    return comparison


def printHeatmap(class_sep, method):
    """
    print heatmap showing class distances
    Parameters
    ----------
    class_sep: dict
        result of calcSep()
    method: str
        has to be "Euclidean", "Manhattan" or "Chebyshev"
    Examples
    --------
    Returns
    -------
    Nothing
    """
    df = pd.DataFrame(class_sep[method])
    sb.heatmap(df, yticklabels=class_sep[method].keys())


def select_samples(path, path_ref_p, ref_p_name, out_folder_resampled_scenes, raster_ext, train_size, random_state,
                   max_depth, n_cores, n_estimator):
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
    gt = ref_p.GetGeoTransform()
    prj = ref_p.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())
    # print(ref_p.shape[0], ref_p.shape[1])
    # print(np.unique(ref_p, return_counts=True))

    print(f"Creating data frame with labels and pixel values from satellite images")

    df = pd.DataFrame()
    labels = pd.Series(np.array(ref_p[:]).flat)  # read the class labels
    df['Label_nr'] = labels

    for i in range(len_ras_li):
        file = gdal.Open(raster_file_list[i])
        file = np.array(file.GetRasterBand(1).ReadAsArray())
        file = file.flatten()
        layer = pd.Series(np.array(file).flat)
        df['file_{}'.format(i + 1)] = layer

    # print(df.shape)
    df2 = df[df != -99]  # remove all -99 values from data frame
    df2 = df2.dropna()  # remove all NaN values from data frame
    # print(df2.shape)
    print(df.head())

    print(f"Removing -99 and NaN-values from data frame")

    print(df2.head())

    # print("trennbarkeit")

    # class_sep = calcSep(df2, "Label_nr")
    # print(class_sep)
    # printHeatmap(class_sep, "Chebyshev")
    # plt.show()

    print(f"Using random sample selection")
    row_count = df2.shape[1]  # get max rows from data frame

    x = df2.iloc[:, 1:row_count].values
    y = df2.iloc[:, 0].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state)
    print(y_train)
    print("np.uniqie")
    print(np.unique(y_train, return_counts=True))

    print(f"{len(x_train)} pixels used for training and {len(x_test)} pixels used for testing \n")

    # y_train = y_train.values.ravel()
    rf = ensemble.RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimator,
                                         n_jobs=n_cores, verbose=2)
    rf.fit(x_train, y_train)
    fi = rf.feature_importances_
    print(fi)
    print(len(fi))

    rf_pred = rf.predict(x_test)
    print(rf_pred)
    data = df.iloc[:, 1:]

    pred_all = rf.predict(data)

    # pred_all.reshape(4457843, 1)
    print(np.unique(pred_all, return_counts=True))
    print(pred_all)
    print(len(pred_all))

    grid = pred_all.reshape((2503, 1781))
    values = np.unique(pred_all.ravel())
    img = np.empty((grid.shape[0], grid.shape[1], 3))
    print(type(grid))
    print(grid.shape)
    print(grid)
    plt.figure(figsize=(10, 10))
    plt.imshow(img, interpolation='none')
    plt.tight_layout()
    # plt.show()
    # plt.savefig("C:/GEO419/Spain_Donana_S1-VV/test/test/pred.png")

    driver = gdal.GetDriverByName('GTIFF')
    rows, cols = ref_p.shape
    fp = "C:/GEO419/results/test_3class.tif"
    out_ds = driver.Create(fp, cols, rows, 1, gdal.GDT_UInt16)
    print(type(out_ds))
    # writting output raster
    out_ds.GetRasterBand(1).WriteArray(grid)
    out_ds.SetGeoTransform(gt)
    # setting spatial reference of output raster
    out_ds.SetProjection(srs.ExportToWkt())
    # Close output raster dataset
    out_ds = None

    pred = pred_all

    return x_train, x_test, y_train, y_test, pred


def rf_old(max_depth, n_estimator, random_state, x_train, x_test, y_train, path,
           out_folder_resampled_scenes, raster_ext):
    """
    The function calls the RandomForrest algorithm from sklearn, builds the tree and calculates the prediction.
    Parameters
    ----------
    max_depth: int
        maximal depth of each tree
    n_estimator: int
        number of estimators
    random_state: int
        Returns a random number between 0 and 43 and ensures that the random forests are not identical in multiple
        executions.
    x_train: pandas DataFrame
        DataFrame with training pixels from satellite images
    x_test: pandas DataFrame
        DataFrame with test pixels from satellite images
    y_train: pandas DataFrame
        DataFrame with training labels from reference product images
    Examples
    --------
    Returns
    -------
    pred
        array with predicted labels
    """
    print(f"Build random forest with {n_estimator} trees and a max depth of {max_depth}")
    rfcb = ensemble.RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimator)
    print(f"Fitting random forest to training data")
    rfcb.fit(x_train, y_train)
    print(f"Predicting results")

    res_path = os.path.join(path, out_folder_resampled_scenes)
    raster_file_list, raster_file_name = geodata.parse_folder(res_path, raster_ext)

    # file = geodata.open_raster_gdal(res_path, raster_file_name[0])
    file = os.path.join(res_path, raster_file_name[0])
    print(file)
    # file = np.array(file.GetRasterBand(1).ReadAsArray())

    img_ds = gdal.Open(file, gdal.GA_ReadOnly)

    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    row = img_ds.RasterYSize
    col = img_ds.RasterXSize
    band_number = img_ds.RasterCount

    print('Image extent: {} x {} (row x col)'.format(row, col))
    print('Number of Bands: {}'.format(band_number))
    print(img.shape[0], img.shape[1], img.shape[2])
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    print(type(new_shape))
    print(type(img.shape[0]), type(img.shape[1]), type(img.shape[2]))
    # img_as_array = file[:, :, :np.int(img.shape[2])].reshape(new_shape)
    img_as_array = np.nan_to_num(new_shape)
    img_as_array.reshape(1, -1)
    class_prediction = rfcb.predict(img_as_array)
    # pred = rfcb.predict(x_test) # hier das gesamte Bild als prediction angeben und nicht testdaten
    #
    # return class_prediction, pred
