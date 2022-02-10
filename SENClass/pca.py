"""
pca.py: In the script, the Principal component analysis will be performed.
@author: Anastasiia Vynohradova
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def principal(data, x_train):
    """
    Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
    Parameters
    ----------
    data
    x_train
    Returns
    -------
    data_transformed
    x_train_transformed
    """
    print('\n####################   -   Performing PCA   -   ####################')
    pca = PCA(n_components=3)
    data = StandardScaler().fit_transform(data)
    data_transformed = pca.fit_transform(data)

    x_train = StandardScaler().fit_transform(x_train)
    x_train = pca.fit_transform(x_train)

    x_train_transformed = pd.DataFrame(x_train)
    data_transformed = pd.DataFrame(data_transformed)

    return data_transformed, x_train_transformed
