import numpy as np
import pytest

from mlxtend.cluster import Kmeans


def _samples(dimensions):
    samples = np.array([[0.0], [3.0], [4.0], [10.0], [11.0]])
    if dimensions == 2:
        samples = np.hstack([samples, 2 * samples])
    return samples


def _assert_final_assignments(model, samples):
    squared_distances = np.sum(
        (samples[:, None, :] - model.centroids_[None, :, :]) ** 2, axis=2
    )
    nearest = np.argmin(squared_distances, axis=1)
    np.testing.assert_array_equal(model.predict(samples), nearest)
    assert set(model.clusters_) == set(range(model.k))
    for cluster in range(model.k):
        np.testing.assert_array_equal(
            model.clusters_[cluster], np.flatnonzero(nearest == cluster)
        )


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("max_iter", [1, 2, 10])
def test_iteration_limit_keeps_cluster_membership_current(dimensions, max_iter):
    samples = _samples(dimensions)
    original = samples.copy()

    model = Kmeans(k=2, max_iter=max_iter, random_seed=0).fit(samples)

    _assert_final_assignments(model, samples)
    np.testing.assert_array_equal(samples, original)
    if max_iter == 1:
        np.testing.assert_allclose(model.centroids_[:, 0], [7, 0])
        assert model.iterations_ == 1
    elif max_iter == 2:
        np.testing.assert_allclose(model.centroids_[:, 0], [25 / 3, 1.5])
        assert model.iterations_ == 2


@pytest.mark.parametrize("dimensions", [1, 2])
def test_continued_single_steps_refresh_cluster_membership(dimensions):
    samples = _samples(dimensions)
    model = Kmeans(k=2, max_iter=1, random_seed=0).fit(samples)
    _assert_final_assignments(model, samples)

    model.fit(samples, init_params=False)

    _assert_final_assignments(model, samples)
    np.testing.assert_allclose(model.centroids_[:, 0], [25 / 3, 1.5])
    assert model.iterations_ == 2


def test_converged_initial_centroids_keep_existing_assignments():
    samples = np.array([[0.0], [10.0]])
    model = Kmeans(k=2, max_iter=10, random_seed=0).fit(samples)

    _assert_final_assignments(model, samples)
    assert model.iterations_ == 0
