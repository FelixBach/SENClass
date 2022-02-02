"""
pca.py: In the script, the Principal component analysis will be performed.
@author: Anastasiia Vynohradova
"""

from sklearn.decomposition import PCA


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
    pca = PCA(n_components=15)
    data_transformed = pca.fit_transform(data)
    x_train_transformed = pca.fit_transform(x_train)

    return data_transformed, x_train_transformed