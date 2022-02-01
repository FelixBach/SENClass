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
    The RandomForest will be created with this function.
    Parameters
    ----------
    max_depth: int
        specifies the maximum depth of the random forest
    random_state: int
        Returns a random number between 0 and 43 and ensures that the randomly selected elements are not identical in
        multiple executions.
    n_estimators: int
        specifies the number of trees in the forest
    n_cores: int
        specifies how many cores are used to fit the model
    verbose: int
        shows progress in console
    Returns
    -------
    sklearn.ensemble._forest.RandomForestClassifier
        unfitted RandomForest model
    """
    rf = ensemble.RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators,
                                         n_jobs=n_cores, verbose=verbose)
    return rf


def rf_fit(rf, x_train, y_train):
    """
    rf_fit will create the Random Forrest with the defined parameters and fit the model to the training data.
    Parameters
    ----------
    rf: sklearn.ensemble._forest.RandomForestClassifier
        RandomForest which will be trained
    x_train: numpy.ndarray
        array with training values (pixel values from satellite)
    y_train: numpy.ndarray
        array with training values (label values)
    Returns
    -------
    sklearn.ensemble._forest.RandomForestClassifier
        fitted RandomForest model
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
    data: pandas.core.frame.DataFrame

    rf_fitted: sklearn.ensemble._forest.RandomForestClassifier
        fitted RandomForest model
    Returns
    -------
    numpy.ndarray
        Array with predicted labels
    """
    data = data.iloc[:, 1:]
    prediction = rf_fitted.predict(data)

    return prediction


def rf_parameter_tuning(x_train, y_train, data, min_depth_t, max_depth_t, min_estimator, max_estimator, value_generator,
                        n_iter, cv, random_state, n_cores):
    """
    The function searches for the best RandomForest parameters and will later fit the best performing model and create
    the prediction.
    Parameters
    ----------
    x_train: numpy.ndarray
        array with training values (pixel values from satellite)
    y_train: numpy.ndarray
        array with training values (label values)
    data: pandas.core.frame.DataFrame
        Values to apply the prediction to
    min_depth_t: int
        minimum depth of the random forest
    max_depth_t: int
        maximum depth of the random forest
    min_estimator: int
        specifies the minimum number of trees in the forest
    max_estimator:  int
        specifies the maximum number of trees in the forest
    value_generator: int
        Generates example values for hyper tuning. As an example, the value of min_estimator is set to 10 and
        max_estimator is set to 20. If value_generator is set to 2, RandomForests are created that have, for example,
        12 or 14 n_estimator. If the value of value_generator is set to 5, RandomForests are created that have, for
        example, 11,12,14,17 and 18 n_estiamtors.
    n_iter: int
        Number of parameter settings that are sampled.
    cv: int
        number of folds of cross validation
    random_state: int
        Returns a random number between 0 and 43 and ensures that the randomly selected elements are not identical in
        multiple executions.
    n_cores: int
        specifies how many cores are used to fit the model
    Returns
    -------
    numpy.ndarray
        Array with predicted labels
    """
    search_grid = {'n_estimators': [int(x) for x in np.linspace(start=min_estimator, stop=max_estimator,
                                                                num=value_generator)],
                   'max_features': ['auto', 'sqrt'],
                   'max_depth': [int(x) for x in np.linspace(start=min_depth_t, stop=max_depth_t,
                                                             num=value_generator)]}

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
    print(f"Best performing RandomForestModel has the following parameters: {best_model} \n")

    data = data.iloc[:, 1:]
    best_model_pred = best_model.predict(data)

    return best_model_pred
