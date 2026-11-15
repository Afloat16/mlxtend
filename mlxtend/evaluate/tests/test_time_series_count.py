import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from mlxtend.evaluate import GroupTimeSeriesSplit

CONFIGS = [
    {"test_size": 1, "train_size": 3},
    {"test_size": 2, "train_size": 3},
    {"test_size": 1, "train_size": 2, "gap_size": 1, "shift_size": 2},
    {"test_size": 2, "train_size": 2, "gap_size": 1, "shift_size": 2},
]


def regression_data(group_kind):
    groups = np.repeat(np.arange(8), np.arange(1, 9))
    if group_kind == "names":
        groups = np.array(["period-" + str(value) for value in groups])
    rng = np.random.RandomState(29)
    samples = rng.normal(size=(len(groups), 2))
    targets = 1.7 * samples[:, 0] + rng.normal(scale=0.2, size=len(groups))
    return samples, targets, groups


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("group_kind", ["numbers", "names"])
def test_split_count_is_available_before_iteration(config, group_kind):
    samples, targets, groups = regression_data(group_kind)
    splitter = GroupTimeSeriesSplit(**config)
    original_attributes = vars(splitter).copy()
    expected = list(GroupTimeSeriesSplit(**config).split(samples, targets, groups))

    assert splitter.get_n_splits(groups=groups) == len(expected)
    assert splitter.get_n_splits(samples, targets, groups) == len(expected)
    assert vars(splitter) == original_attributes
    actual = list(splitter.split(samples, targets, groups))
    assert splitter.get_n_splits() == len(expected)
    for observed, reference in zip(actual, expected):
        np.testing.assert_array_equal(observed[0], reference[0])
        np.testing.assert_array_equal(observed[1], reference[1])


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("group_kind", ["numbers", "names"])
def test_grid_search_matches_explicit_time_series_splits(config, group_kind):
    samples, targets, groups = regression_data(group_kind)
    splits = list(GroupTimeSeriesSplit(**config).split(samples, targets, groups))
    kwargs = {
        "estimator": Ridge(),
        "param_grid": {"alpha": [0.1, 1.0, 10.0]},
        "scoring": "neg_mean_squared_error",
    }
    reference = GridSearchCV(cv=splits, **kwargs).fit(samples, targets)
    actual = GridSearchCV(cv=GroupTimeSeriesSplit(**config), **kwargs).fit(
        samples, targets, groups=groups
    )

    assert actual.n_splits_ == len(splits)
    assert actual.best_params_ == reference.best_params_
    for key, expected in reference.cv_results_.items():
        if key.endswith("_test_score"):
            np.testing.assert_allclose(actual.cv_results_[key], expected)
    np.testing.assert_allclose(actual.predict(samples), reference.predict(samples))


def test_count_queries_do_not_cache_inferred_values():
    splitter = GroupTimeSeriesSplit(test_size=1, train_size=2)
    for size in (5, 9, 4):
        groups = np.repeat(np.arange(size), 2)
        assert splitter.get_n_splits(groups=groups) == size - 2
        assert splitter.n_splits is None
        assert splitter.train_size == 2


@pytest.mark.parametrize("window_type", ["rolling", "expanding"])
def test_explicit_count_does_not_require_groups(window_type):
    splitter = GroupTimeSeriesSplit(test_size=1, n_splits=3, window_type=window_type)
    assert splitter.get_n_splits() == 3


@pytest.mark.parametrize("group_kind", ["numbers", "names"])
def test_count_rejects_nonconsecutive_groups(group_kind):
    groups = [0, 1, 0, 2]
    if group_kind == "names":
        groups = ["period-" + str(value) for value in groups]
    splitter = GroupTimeSeriesSplit(test_size=1, train_size=2)
    with pytest.raises(ValueError, match="The groups should be consecutive"):
        splitter.get_n_splits(groups=groups)


def test_inferred_count_requires_groups():
    splitter = GroupTimeSeriesSplit(test_size=1, train_size=2)
    with pytest.raises(ValueError, match="The groups should be specified"):
        splitter.get_n_splits()


def test_count_rejects_insufficient_groups_without_mutation():
    splitter = GroupTimeSeriesSplit(test_size=2, train_size=4)
    with pytest.raises(ValueError, match="Not enough data"):
        splitter.get_n_splits(groups=np.arange(3))
    assert splitter.n_splits is None
    assert splitter.get_n_splits(groups=np.arange(8)) == 3
