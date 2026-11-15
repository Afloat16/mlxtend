import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.decomposition import KernelPCA

from mlxtend.feature_extraction import RBFKernelPCA

X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 3.0], [4.0, 1.0]])
QUERIES = np.array([[0.5, 1.0], [3.0, 2.0], [-2.0, 4.0]])


@pytest.mark.parametrize("n_components", [None, 1, 2, 3])
def test_transform_training_samples_uses_fitted_projection(n_components):
    model = RBFKernelPCA(gamma=0.2, n_components=n_components).fit(X)
    projected = model.transform(X)
    assert projected.shape == model.X_projected_.shape
    assert_allclose(projected, model.X_projected_, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("gamma", [0.02, 0.2])
def test_held_out_projection_matches_independent_kernel_pca(gamma):
    model = RBFKernelPCA(gamma=gamma, n_components=2).fit(X)
    reference = KernelPCA(kernel="rbf", gamma=gamma, n_components=2)
    reference_training = reference.fit_transform(X)
    scale = np.sqrt(reference.eigenvalues_)
    reference_axes = reference_training / scale
    signs = np.sign(np.sum(model.X_projected_ * reference_axes, axis=0))
    assert_allclose(model.X_projected_, reference_axes * signs, atol=1e-12)
    expected = reference.transform(QUERIES) / scale * signs
    assert_allclose(model.transform(QUERIES), expected, rtol=1e-11, atol=1e-12)


def test_projection_is_independent_of_query_batch():
    model = RBFKernelPCA(gamma=0.2, n_components=2).fit(X)
    individual = np.vstack([model.transform(row[None, :]) for row in QUERIES])
    assert_allclose(model.transform(QUERIES), individual, atol=1e-12)


def test_refit_uses_the_new_training_population():
    replacement = np.vstack([X * 2 + 1, [[-1.0, 3.0], [6.0, 2.0]]])
    model = RBFKernelPCA(gamma=0.2, n_components=2).fit(X)
    model.fit(replacement)
    fresh = RBFKernelPCA(gamma=0.2, n_components=2).fit(replacement)
    assert_allclose(model.transform(QUERIES), fresh.transform(QUERIES), atol=1e-12)
