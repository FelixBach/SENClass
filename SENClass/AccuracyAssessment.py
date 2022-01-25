"""
AccuracyAssessment.py: contains functions to assess the accuracy of the RF classifier. The following metrics are evaluated:
    - Confusion matrix (CM)
    - Kappa statistic (Kappa)
@author: Anastasiia Vynohradova
"""
import os
import gdal
import numpy as np
import seaborn as sns
from sklearn import metrics as metrics
from sklearn.metrics import confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa


# Function to create fancy confusion matrix
def get_confusion_matrix(prediction, path_ref_p, ref_p_name):
    """
    The function calls the confusion_matrix from sklearn.metrics
    Parameters
    ----------
    y_test: pandas.Series
        list with test values
    y_pred: numpy.ndarray
        array with the predicted labels
    Returns
    -------
    cf_matrix: numpy.ndarray
        array with confusion matrix
    """
    ref_p_file = os.path.join(path_ref_p, ref_p_name)
    ref_p = gdal.Open(ref_p_file)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())
    ref_p = ref_p.flatten()
    cf_matrix = confusion_matrix(ref_p, prediction)
    return cf_matrix


def plot_confusion_matrix(cf_matrix):
    """
    The function visualizes the confusion matrix as heatmap from seaborn package
    Parameters
    ----------
    cf_matrix: numpy.ndarray
        array with confusion matrix
    Returns
    -------
    cf_heatmap: matplotlib.axes._subplots.AxesSubplot
        cf_heatmap plot with confusion matrix
    """
    cf_heatmap = sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                             fmt='.2%', cmap='Blues')
    return cf_heatmap


def get_kappa(cf_matrix):
    """
    The function calculates a Kappa coefficient
    Parameters
    ----------
    cf_matrix: numpy.ndarray
        array with confusion matrix
    Returns
    -------
    res_kappa: str
        str with the Kappa Coefficient
    """
    kappa = cohens_kappa(cf_matrix).kappa
    res_kappa = f'Kappa Coefficient is {kappa}'
    return print(res_kappa)


def accuracy(prediction, path_ref_p, ref_p_name):
    """
    The function calculates the overall accuracy
    ----------
    pred: array
        array with the predicted labels
    y_test: array
        array with the values from the reclassified clc mask
    Examples
    --------
    Returns
    -------
    """
    ref_p_file = os.path.join(path_ref_p, ref_p_name)
    ref_p = gdal.Open(ref_p_file)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())
    ref_p = ref_p.flatten()
    acc = metrics.accuracy_score(ref_p, prediction)
    res = f'Overall accuracy is {acc}'

    return print(res)
