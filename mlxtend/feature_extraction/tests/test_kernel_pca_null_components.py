"""Zero-variance kernel components must not become arbitrary or infinite features."""

import numpy as np
import pytest
from sklearn.datasets import make_moons

from mlxtend.feature_extraction import RBFKernelPCA


@pytest.mark.parametrize("rows", [2, 4, 7])
@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("gamma", [0.5, 15.0])
def test_constant_training_data_has_zero_projections(rows, components, gamma):
    x = np.full((rows, 2), 3.0)
    model = RBFKernelPCA(gamma=gamma, n_components=components).fit(x)
    # A constant RBF kernel has zero centered variance in every direction.
    np.testing.assert_array_equal(model.X_projected_, np.zeros((rows, components)))
    queries = np.array([[3.0, 3.0], [1.0, 2.0], [8.0, -4.0]])
    with np.errstate(divide="raise", invalid="raise"):
        actual = model.transform(queries)
    np.testing.assert_array_equal(actual, np.zeros((len(queries), components)))
    # Keep the full eigensystem valid even though projected null axes are zero.
    np.testing.assert_allclose(
        model.e_vecs_.T @ model.e_vecs_, np.eye(rows), atol=1e-14
    )


def test_single_training_sample_has_no_kernel_variance():
    model = RBFKernelPCA(n_components=1).fit(np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(model.X_projected_, [[0.0]])
    with np.errstate(divide="raise", invalid="raise"):
        actual = model.transform(np.array([[1.0, 2.0], [4.0, 5.0]]))
    np.testing.assert_array_equal(actual, [[0.0], [0.0]])


@pytest.mark.parametrize("gamma", [1.0, 1e-6])
def test_duplicate_groups_keep_only_the_nonzero_kernel_direction(gamma):
    x = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    model = RBFKernelPCA(gamma=gamma, n_components=2).fit(x)
    # The two equally sized groups have one centered contrast. Its normalized
    # sample coordinates are +/- 1/2, and every other direction has zero variance.
    np.testing.assert_allclose(np.abs(model.X_projected_[:, 0]), 0.5, atol=1e-10)
    np.testing.assert_array_equal(model.X_projected_[:, 1], np.zeros(len(x)))
    with np.errstate(divide="raise", invalid="raise"):
        transformed = model.transform(x)
    np.testing.assert_allclose(transformed, model.X_projected_, atol=1e-10)


def test_nonzero_components_retain_the_original_eigenvectors():
    x, _ = make_moons(n_samples=20, random_state=7)
    model = RBFKernelPCA(gamma=15, n_components=2).fit(x)
    np.testing.assert_array_equal(model.X_projected_, model.e_vecs_[:, :2])
    assert np.isfinite(model.transform(x)).all()
