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
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from statsmodels.stats.inter_rater import cohens_kappa


# Function to create fancy confusion matrix
def get_confusion_matrix(prediction, out_ref_p):
    """
    The function calls the confusion_matrix from sklearn.metrics
    Parameters
    ----------
    prediction: numpy.ndarray
        array with the predicted labels
    out_ref_p: string
        path to resampled/reclassified reference product
    Returns
    -------
    cf_matrix: numpy.ndarray
        array with confusion matrix
    """
    ref_p = gdal.Open(out_ref_p)
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


def accuracy(prediction, out_ref_p):
    """
        The function calculates the overall accuracy
    Parameters
    ----------
    prediction: numpy.ndarray
        array with the predicted labels
    out_ref_p:
        path to resampled/reclassified reference product
    Returns
    -------

    """
    ref_p = gdal.Open(out_ref_p)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())
    ref_p = ref_p.flatten()
    acc = metrics.accuracy_score(ref_p, prediction)
    res = f'Overall accuracy is {acc}'
    balanced_acc = balanced_accuracy_score(ref_p, prediction)
    balanced_acc = f'Balanced accuracy is {balanced_acc}'

    return print(res), print(balanced_acc)
