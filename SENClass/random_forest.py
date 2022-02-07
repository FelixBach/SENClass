"""
random_forest.py: In the script, the random forest is created, fited to the data, and the labels are predicted.
@author: Felix Bachmann
"""

import gdal
import numpy as np
import warnings
import pandas as pd
from sklearn import ensemble as ensemble
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics as metrics


warnings.simplefilter("ignore", UserWarning)


def rf_fit(max_depth, random_state, n_estimators, n_cores, verbose, x_train, y_train):
    """
    rf_fit will create the Random Forrest with the defined parameters and fit the model to the training data.
    Parameters
    ----------
    max_depth
    random_state
    n_estimators
    n_cores
    verbose
    x_train
    y_train
    Returns
    -------
    sklearn.ensemble._forest.RandomForestClassifier
    """
    rf = ensemble.RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators,
                                         n_jobs=n_cores, verbose=verbose)
    rf_fitted = rf.fit(x_train, y_train)

    return rf, rf_fitted


def rf_feature_importance(rf):
    """
    Parameters
    ----------
    rf
    Returns
    -------
    numpy.ndarray
    """
    feature_importance = rf.feature_importances_

    return feature_importance


def rf_predict(data, rf_fitted):
    """
    Parameters
    ----------
    data
    rf_fitted
    Returns
    -------
    numpy.ndarray
    """
    data = data.iloc[:, 1:]
    prediction = rf_fitted.predict(data)

    return prediction


def rf_feature_selection(rf, x_train, y_train, prediction, out_ref_p, data):
    """
        Select important features and calculate new RF model
        Parameters
        ----------
        rf: sklearn.RandomForestClassifier object
            Random Forest Model the selection is based on
        x_train: list
            training values dataset
        y_train: list
            training labels dataset
        x_test: list
            test values dataset
        Examples
        --------
        >>> x_important_train, x_important_test =
            selectImportantFeatures(base_model, x_train, y_train, x_test)
        Returns
        -------
        list
            new training datasets containing only values for selected features
        """
    sel = SelectFromModel(rf)
    sel.fit(x_train, y_train)
    x_train = pd.DataFrame(x_train)
    selected_feat = x_train.columns[(sel.get_support())]
    print(str(len(selected_feat)) + " features selected")
    print(selected_feat)
    x_important_train = sel.transform(x_train)

    rf = ensemble.RandomForestClassifier(max_depth=2, random_state=0, n_estimators=3,
                                         n_jobs=-1, verbose=2)
    rf_fitted = rf.fit(x_important_train, y_train)

    data = data.iloc[:, 1:]
    prediction = rf_fitted.predict(data)


    ref_p = gdal.Open(out_ref_p)
    ref_p = np.array(ref_p.GetRasterBand(1).ReadAsArray())
    ref_p = ref_p.flatten()
    acc = metrics.accuracy_score(ref_p, prediction)
    res = f'Overall accuracy is {acc}'
    print(res)
