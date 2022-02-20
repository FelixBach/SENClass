"""
pca.py: In the script, the Principal component analysis will be performed.
@author: Anastasiia Vynohradova
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def principal(data, x_train, n_components):
    """
    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Values to apply the prediction to
    x_train: numpy.ndarray
        array with training values (pixel values from satellite)
    n_components: int
        Number of dimensions the data will be reduced to
    Returns
    -------
    data_transformed
    x_train_transformed
    """
    print('\n####################   -   Performing PCA   -   ####################')
    pca = PCA(n_components=n_components)
    data = StandardScaler().fit_transform(data)
    data_transformed = pca.fit_transform(data)

    x_train = StandardScaler().fit_transform(x_train)
    x_train = pca.fit_transform(x_train)

    x_train_transformed = pd.DataFrame(x_train)
    data_transformed = pd.DataFrame(data_transformed)

    return data_transformed, x_train_transformed
