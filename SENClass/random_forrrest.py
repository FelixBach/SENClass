"""
random_forrest.py:
@author: Felix Bachmann
"""

from sklearn import ensemble as ensemble


def rf_classifier_basic(max_depth, n_estimator):
    """
    The function calls the RandomForrest aglogrithm from sklearn
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
        n_estimator=n_estimator
    )
    return rfcb
