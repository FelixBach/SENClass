"""
random_forest.py: In the script, the random forest is created, fited to the data, and the labels are predicted.
@author: Felix Bachmann
"""

from sklearn import ensemble as ensemble
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import matplotlib as mpl


def random_forest_classifier(max_depth, n_estimator, random_state, x_train, x_test, y_train):
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
    pred = rfcb.predict(x_test)

    return pred
