# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import importlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from mlxtend.plotting import plot_decision_regions  # noqa: E402

decision_module = importlib.import_module("mlxtend.plotting.decision_regions")


@pytest.fixture
def pools(monkeypatch):
    original = decision_module.mp.Pool
    created = []

    def tracked_pool(*args, **kwargs):
        pool = original(*args, **kwargs)
        created.append(pool)
        return pool

    monkeypatch.setattr(decision_module.mp, "Pool", tracked_pool)
    yield created
    for pool in created:
        pool.terminate()
        pool.join()


@pytest.fixture
def axes():
    fig, ax = plt.subplots(figsize=(0.4, 0.3), dpi=20)
    yield ax
    plt.close(fig)


def _data(dtype=float):
    X = np.array([[-2, -1], [-1, 2], [0, -2], [1, 2], [2, -1], [3, 2]], dtype=dtype)
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y, DecisionTreeClassifier(random_state=1).fit(X, y)


def _assert_grid_matches_serial(ax, X, y, classifier, n_jobs, monkeypatch):
    captured = {}
    original = ax.contourf

    def record_grid(x_grid, y_grid, prediction, **kwargs):
        captured.update(x=x_grid.copy(), y=y_grid.copy(), z=prediction.copy())
        return original(x_grid, y_grid, prediction, **kwargs)

    monkeypatch.setattr(ax, "contourf", record_grid)
    result = plot_decision_regions(X, y, classifier, ax=ax, n_jobs=n_jobs, legend=0)
    assert result is ax
    grid = np.column_stack([captured["x"].ravel(), captured["y"].ravel()]).astype(
        X.dtype
    )
    expected = classifier.predict(grid).reshape(captured["z"].shape)
    np.testing.assert_array_equal(captured["z"], expected)
    ax.figure.canvas.draw()


@pytest.mark.parametrize("n_jobs, available", [(-1, 1), (-1, 2), (2, 2), (3, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_parallel_grid_matches_serial_predictions(
    pools, axes, monkeypatch, n_jobs, available, dtype
):
    monkeypatch.setattr(decision_module.mp, "cpu_count", lambda: available)
    X, y, classifier = _data(dtype)
    before = X.copy(), y.copy()
    _assert_grid_matches_serial(axes, X, y, classifier, n_jobs, monkeypatch)
    np.testing.assert_array_equal(X, before[0])
    np.testing.assert_array_equal(y, before[1])


@pytest.mark.parametrize("n_jobs", [-1, 8])
def test_more_workers_than_grid_points_does_not_predict_empty_batches(
    pools, axes, monkeypatch, n_jobs
):
    monkeypatch.setattr(decision_module.mp, "cpu_count", lambda: 8)
    axes.figure.set_size_inches(0.1, 0.1)
    X, y, classifier = _data()
    _assert_grid_matches_serial(axes, X, y, classifier, n_jobs, monkeypatch)


@pytest.mark.parametrize("n_jobs", [-1, 2])
def test_prediction_failure_releases_worker_processes(pools, axes, monkeypatch, n_jobs):
    monkeypatch.setattr(decision_module.mp, "cpu_count", lambda: 2)
    X, y, _ = _data()
    wrong_shape_model = DecisionTreeClassifier().fit(X[:, :1], y)
    with pytest.raises(ValueError, match="features"):
        plot_decision_regions(X, y, wrong_shape_model, ax=axes, n_jobs=n_jobs)
    assert pools
    assert all(not worker.is_alive() for pool in pools for worker in pool._pool)


def test_task_serialization_failure_releases_worker_processes(pools, axes, monkeypatch):
    monkeypatch.setattr(decision_module.mp, "cpu_count", lambda: 2)
    X, y, classifier = _data()
    classifier.unpickleable = lambda value: value
    with pytest.raises(AttributeError, match="local object|pickle"):
        plot_decision_regions(X, y, classifier, ax=axes, n_jobs=2)
    assert pools
    assert all(not worker.is_alive() for pool in pools for worker in pool._pool)
