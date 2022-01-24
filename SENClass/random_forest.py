"""
random_forest.py: In the script, the random forest is created, fited to the data, and the labels are predicted.
@author: Felix Bachmann
"""

import warnings
from sklearn import ensemble as ensemble

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
