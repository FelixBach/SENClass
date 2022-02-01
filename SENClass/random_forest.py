"""
random_forest.py: In the script, the random forest is created, fitted to the data, and the labels are predicted.
@author: Felix Bachmann
"""

import gdal
import numpy as np
import pandas as pd
from sklearn import ensemble as ensemble
from sklearn.model_selection import RandomizedSearchCV
import warnings


warnings.simplefilter("ignore", UserWarning)


def rf_create(max_depth, random_state, n_estimators, n_cores, verbose):
    """
    Parameters
    ----------
    max_depth:

    random_state:

    n_estimators:

    n_cores:

    verbose:

    Returns
    -------

    """
    rf = ensemble.RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators,
                                         n_jobs=n_cores, verbose=verbose)
    return rf


def rf_fit(rf, x_train, y_train):
    """
    rf_fit will create the Random Forrest with the defined parameters and fit the model to the training data.
    Parameters
    ----------
    rf:

    x_train:

    y_train:

    Returns
    -------
    sklearn.ensemble._forest.RandomForestClassifier
    """
    rf = rf
    rf_fitted = rf.fit(x_train, y_train)

    return rf_fitted


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
    """
    data = data.iloc[:, 1:]
    prediction = rf_fitted.predict(data)

    return prediction


def rf_parameter_tuning(x_train, y_train, data, min_depth_t, max_depth_t, min_estimator, max_estimator, number_estimator,
                        n_iter, cv, random_state, n_cores):
    """
    Parameters
    ----------
    x_train:

    y_train:

    data:

    min_depth_t: int
        minimum depth of the random forest
    max_depth_t: int
        maximum depth of the random forest
    min_estimator: int

    max_estimator:  int

    number_estimator: int

    n_iter: int

    cv: int

    random_state:

    n_cores:

    Returns
    -------
    numpy.ndarray
        Array with predicted labels
    """
    search_grid = {'n_estimators': [int(x) for x in np.linspace(start=min_estimator, stop=max_estimator,
                                                                num=number_estimator)],
                   'max_features': ['auto', 'sqrt'],
                   'max_depth': [int(x) for x in np.linspace(start=min_depth_t, stop=max_depth_t,
                                                             num=number_estimator)]}

    tune_model = ensemble.RandomForestClassifier()
    tune_model_grid = RandomizedSearchCV(
        estimator=tune_model,
        param_distributions=search_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=n_cores)

    tune_model_grid.fit(x_train, y_train)
    best_model = tune_model_grid.best_estimator_
    print(f"Best performing RandomForestModel has the following parameters: {best_model}")

    data = data.iloc[:, 1:]
    best_model_pred = best_model.predict(data)
    print(type(best_model_pred))

    return best_model_pred
