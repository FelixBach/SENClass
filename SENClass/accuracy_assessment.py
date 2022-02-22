"""
accuracy_assessment.py: contains functions to assess the accuracy of the RF classifier. The following metrics are
evaluated:
    - Confusion matrix (CM)
    - Kappa statistic (Kappa)
@author: Anastasiia Vynohradova
"""
import gdal
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from statsmodels.stats.inter_rater import cohens_kappa
from sklearn.metrics import classification_report


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


def accuracy_assessment(prediction, out_ref_p):
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
    This function has no return
    """
    # open reference product
    print('\n####################   -   Accuracy Assessment   -   ####################')
    ref_p = gdal.Open(out_ref_p)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())
    ref_p = ref_p.flatten()

    cf_matrix_pd = pd.crosstab(ref_p, prediction, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print('\n##########   -   Confusion Matrix   -   ##########')

    # display full dataframe without truncation
    with pd.option_context('display.max_columns', 100,
                           'display.width', 640):
        print(cf_matrix_pd)

    cf_matrix = confusion_matrix(ref_p, prediction, normalize=None)

    class_report = classification_report(ref_p, prediction)
    print('\n##########   -   Classification report   -   ##########')
    print(class_report)
    get_kappa(cf_matrix)
