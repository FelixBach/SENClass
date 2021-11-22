"""
random_forrest.py:
@author: Felix Bachmann
"""

import numpy as np
from sklearn import ensemble as ensemble
from sklearn import metrics as metrics


def rf_classifier_basic(max_depth, n_estimator):
    """
    The function calls the RandomForrest algorithm from sklearn
    ----------
    max_depth: int
        maximal depth of each tree
    n_estimators: int
        number of estimators
    Examples
    --------
    Returns
    -------
    sklearn.RandomForestClassifier object
        Random Forest model
    """
    rfcb = ensemble.RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimator
    )
    return rfcb


def rf_classifier_fit(rfcb, x_train, y_train):
    """
    The function fits the basic model to the values
    ----------
    rfcb: sklearn.RandomForestClassifier object
        basic rf model that will be fitted
    x_train: list
        list with training values
    y_train: list
        list with testing values
    Examples
    --------
    Returns
    -------
    sklearn.RandomForestClassifier object
        fitted Random Forest model
    """
    rfcb_fit = rfcb.fit(x_train, y_train)

    return rfcb_fit


def rf_calssifier_predict(rfcb_fit, x_test):
    """
    The function predicts the class values
    ----------
    rfcb_fit: sklearn.RandomForestClassifier object
        fitted random forrest model
    x_test: list
        list with test values
    Examples
    --------
    Returns
    -------
    sklearn.RandomForestClassifier object
        Random Forest model
    """

    return rfcb_fit.predict(x_test)


def accu(pred, y_test):
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
    sklearn.RandomForestClassifier object
        Random Forest model
    """
    acc = metrics.accuracy_score(y_test, pred)
    res = f'Overall accuracy is {acc}'

    return print(res)
