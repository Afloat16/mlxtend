# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from mlxtend.cluster import Kmeans


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_predictions_match_independent_distance_kernel(dtype, strided):
    rng = np.random.RandomState(137)
    values = rng.randint(-5, 6, size=(12003, 12)).astype(dtype)
    X = values[:, ::2] if strided else values[:, :6].copy()
    centroids = rng.randint(-5, 6, size=(7, 6)).astype(dtype)
    original = values.copy()
    original_centroids = centroids.copy()
    model = Kmeans(k=len(centroids))
    model.centroids_ = centroids
    model._is_fitted = True

    # scipy's independently implemented distance kernel also checks first-index
    # tie handling for these exactly represented integer-valued coordinates.
    expected = cdist(X, centroids, metric="sqeuclidean").argmin(axis=1)
    np.testing.assert_array_equal(model.predict(X), expected)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(centroids, original_centroids)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_duplicate_centroids_and_exact_ties_keep_first_label(dtype):
    centroids = np.array([[-2, 0], [2, 0], [-2, 0]], dtype=dtype)
    X = np.tile(np.array([[0, 0], [-2, 0], [2, 0], [1, 0]], dtype=dtype), (5001, 1))
    model = Kmeans(k=3)
    model.centroids_ = centroids.copy()
    model._is_fitted = True
    np.testing.assert_array_equal(model.predict(X), np.tile([0, 0, 1, 1], 5001))


def _lloyd_reference(X, k, seed, max_iter, tolerance):
    indices = np.random.RandomState(seed).choice(len(X), k, replace=False)
    centroids = X[indices].copy()
    iterations = 0
    for _ in range(max_iter):
        assignments = cdist(X, centroids, metric="sqeuclidean").argmin(axis=1)
        assert all(np.any(assignments == label) for label in range(k))
        updated = np.array([X[assignments == label].mean(axis=0) for label in range(k)])
        if np.allclose(centroids, updated, rtol=tolerance, atol=1e-8):
            break
        centroids = updated
        iterations += 1
    return centroids, iterations


@pytest.mark.parametrize("max_iter", [1, 3, 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_full_fit_matches_independent_lloyd_assignments(max_iter, dtype):
    rng = np.random.RandomState(12)
    X = rng.normal(size=(8001, 4)).astype(dtype)
    X += np.repeat([[-4, 0, 0, 0], [0, 4, 0, 0], [4, 0, 0, 0]], 2667, axis=0)
    original = X.copy()
    expected, iterations = _lloyd_reference(X, 3, 2, max_iter, 1e-5)
    model = Kmeans(k=3, random_seed=2, max_iter=max_iter).fit(X)
    np.testing.assert_allclose(model.centroids_, expected, rtol=1e-6, atol=1e-6)
    assert model.iterations_ == iterations
    np.testing.assert_array_equal(
        model.predict(X), cdist(X, expected, metric="sqeuclidean").argmin(axis=1)
    )
    np.testing.assert_array_equal(X, original)


def test_empty_prediction_batch_and_single_centroid():
    model = Kmeans(k=1, random_seed=0).fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert model.predict(np.empty((0, 2))).shape == (0,)
    np.testing.assert_array_equal(model.predict(np.ones((10000, 2))), 0)
