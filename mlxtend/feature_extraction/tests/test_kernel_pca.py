# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from sklearn.datasets import make_moons

from mlxtend.feature_extraction import RBFKernelPCA as KPCA

X1, y1 = make_moons(n_samples=50, random_state=1)


def test_default_components():
    pca = KPCA()
    pca.fit(X1)
    assert pca.X_projected_.shape == X1.shape


def test_default_2components():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    assert pca.X_projected_.shape == (X1.shape[0], 2)


def test_default_0components():
    with pytest.raises(AttributeError):
        pca = KPCA(n_components=0)
        pca.fit(X1)


def test_proj():
    pca = KPCA(n_components=2)
    pca.fit(X1[:2])
    exp = np.array([[-0.71, -0.71], [0.71, -0.71]])
    assert_almost_equal(pca.X_projected_, exp, decimal=2)


def test_reproj_1():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    exp = pca.transform(X1)
    assert_almost_equal(pca.X_projected_, exp, decimal=2)


def test_reproj_2():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    exp = pca.transform(X1[1, None])
    assert_almost_equal(pca.X_projected_[1, None], exp, decimal=2)


def test_fail_array_fit():
    pca = KPCA(n_components=2)
    with pytest.raises(ValueError):
        pca.fit(X1[1])


def test_fail_array_transform():
    pca = KPCA(n_components=2)
    pca.fit(X1)
    with pytest.raises(ValueError):
        pca.transform(X1[1])


@pytest.mark.parametrize("n_components", [None, 2, 10])
def test_transform_preserves_fitted_projection(n_components):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [4.0, 1.0], [8.0, 8.0]])
    pca = KPCA(n_components=n_components, gamma=0.1).fit(X)

    actual = pca.transform(X)

    assert actual.shape == pca.X_projected_.shape
    assert_allclose(actual, pca.X_projected_, atol=1e-12)


def test_transform_uses_training_kernel_center_for_new_samples():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [4.0, 1.0], [8.0, 8.0]])
    queries = np.array([[0.5, 1.5], [10.0, 2.0], [-1.0, 4.0]])
    pca = KPCA(n_components=2, gamma=0.1).fit(X)
    training_kernel = np.exp(-0.1 * ((X[:, None] - X[None, :]) ** 2).sum(axis=2))
    query_kernel = np.exp(-0.1 * ((queries[:, None] - X[None, :]) ** 2).sum(axis=2))
    centered = (
        query_kernel
        - query_kernel.mean(axis=1, keepdims=True)
        - training_kernel.mean(axis=0)
        + training_kernel.mean()
    )
    expected = centered @ (pca.X_projected_ / pca.e_vals_[:2])

    assert_allclose(pca.transform(queries), expected, atol=1e-12)


def test_transform_center_is_independent_of_query_batch():
    pca = KPCA(n_components=2, gamma=0.1).fit(X1)
    queries = np.array([[0.5, 1.5], [10.0, 2.0], [-1.0, 4.0]])
    together = pca.transform(queries)
    separately = np.vstack([pca.transform(row[None, :]) for row in queries])

    assert_allclose(together, separately, atol=1e-12)


def test_refit_refreshes_kernel_center():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [4.0, 1.0], [8.0, 8.0]])
    pca = KPCA(n_components=2, gamma=0.1).fit(X1)
    pca.fit(X)

    assert_allclose(pca.transform(X), pca.X_projected_, atol=1e-12)
