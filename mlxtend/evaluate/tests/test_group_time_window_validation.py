# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest

from mlxtend.evaluate import GroupTimeSeriesSplit


@pytest.mark.parametrize("window", ["rolling", "expanding"])
def test_negative_gap_cannot_leak_training_groups_into_test(window):
    with pytest.raises(ValueError, match="gap_size"):
        splitter = GroupTimeSeriesSplit(
            test_size=2, n_splits=3, gap_size=-1, window_type=window
        )
        groups = np.repeat(np.arange(10), 2)
        list(splitter.split(np.zeros((len(groups), 1)), groups=groups))


@pytest.mark.parametrize("n_splits", [None, 3])
def test_zero_shift_cannot_repeat_folds_or_divide_by_zero(n_splits):
    with pytest.raises(ValueError, match="shift_size"):
        splitter = GroupTimeSeriesSplit(
            test_size=1, train_size=2, shift_size=0, n_splits=n_splits
        )
        list(splitter.split(np.zeros((10, 1)), groups=np.arange(10)))


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("test_size", 0),
        ("test_size", -1),
        ("train_size", 0),
        ("train_size", -1),
        ("n_splits", 0),
        ("n_splits", -1),
        ("shift_size", -1),
        ("test_size", 1.5),
        ("gap_size", 0.5),
        ("test_size", None),
    ],
)
def test_invalid_windows_fail_before_producing_any_split(parameter, value):
    params = {"test_size": 1, "train_size": 3, "n_splits": 2}
    params[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        splitter = GroupTimeSeriesSplit(**params)
        list(splitter.split(np.zeros((12, 1)), groups=np.arange(12)))


@pytest.mark.parametrize(
    "params",
    [
        {"test_size": 2, "train_size": 3},
        {"test_size": 2, "train_size": 3, "gap_size": 2, "shift_size": 2},
        {
            "test_size": 1,
            "n_splits": 4,
            "gap_size": 1,
            "shift_size": 2,
            "window_type": "expanding",
        },
        {
            "test_size": 2,
            "train_size": 4,
            "n_splits": 2,
            "gap_size": 1,
            "shift_size": 2,
        },
        {"test_size": 2, "n_splits": 1},
        {"test_size": 2, "n_splits": 1, "window_type": "expanding"},
    ],
)
@pytest.mark.parametrize("integer_type", [int, np.int64, np.uint64])
def test_valid_windows_match_independent_group_boundaries(params, integer_type):
    params = {
        name: integer_type(value) if isinstance(value, int) else value
        for name, value in params.items()
    }
    sizes = [1, 3, 2, 4, 1, 2, 3, 1, 4, 2, 1, 3]
    group_ids = np.repeat(np.arange(12), sizes)
    groups = np.array([f"group_{value}" for value in group_ids])
    X = np.arange(len(groups) * 2).reshape(-1, 2)
    before = X.copy(), groups.copy()
    test_size = int(params["test_size"])
    gap = int(params.get("gap_size", 0))
    shift = int(params.get("shift_size", 1))
    train_size = params.get("train_size")
    count = params.get("n_splits")
    train_size = int(train_size) if train_size is not None else None
    count = int(count) if count is not None else None
    if train_size is None:
        train_size = 12 - gap - test_size - (count - 1) * shift
    if count is None:
        count = (12 - train_size - gap - test_size) // shift + 1
    initial_start = 12 - train_size - gap - test_size - (count - 1) * shift
    splits = list(GroupTimeSeriesSplit(**params).split(X, groups=groups))
    assert len(splits) == count
    for fold, (train, test) in enumerate(splits):
        end = initial_start + train_size + fold * shift
        start = initial_start
        if params.get("window_type", "rolling") == "rolling":
            start += fold * shift
        expected_train = np.flatnonzero((group_ids >= start) & (group_ids < end))
        expected_test = np.flatnonzero(
            (group_ids >= end + gap) & (group_ids < end + gap + test_size)
        )
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)
        assert set(group_ids[train]).isdisjoint(group_ids[test])
        assert group_ids[test[0]] - group_ids[train[-1]] - 1 == gap
    np.testing.assert_array_equal(X, before[0])
    np.testing.assert_array_equal(groups, before[1])


@pytest.mark.parametrize(
    "extra",
    [
        {"n_splits": np.uint64(2**63 + 1)},
        {"n_splits": np.int64(3), "shift_size": np.int64(2**62)},
        {"n_splits": np.uint64(2), "gap_size": np.uint64(2**64 - 1)},
    ],
)
def test_large_numpy_sizes_report_insufficient_data_without_wrapping(extra):
    splitter = GroupTimeSeriesSplit(test_size=1, **extra)
    with np.errstate(over="raise"), pytest.raises(ValueError, match="Not enough data"):
        next(splitter.split(np.zeros((12, 1)), groups=np.arange(12)))
