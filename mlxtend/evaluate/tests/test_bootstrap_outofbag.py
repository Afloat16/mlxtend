# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mlxtend.evaluate import BootstrapOutOfBag
from mlxtend.utils import assert_raises


def test_defaults():
    oob = BootstrapOutOfBag()
    results = list(oob.split(np.array([1, 2, 3, 4, 5])))
    assert len(results) == 200


def test_splits():
    oob = BootstrapOutOfBag(n_splits=3, random_seed=123)
    results = list(oob.split(np.array([1, 2, 3, 4, 5])))
    assert len(results) == 3
    assert np.array_equal(results[0][0], np.array([2, 4, 2, 1, 3]))
    assert np.array_equal(results[0][1], np.array([0]))
    assert np.array_equal(results[-1][0], np.array([1, 1, 0, 0, 1]))
    assert np.array_equal(results[-1][1], np.array([2, 3, 4]))


def test_invalid_splits():
    assert_raises(
        ValueError, "Number of splits must be greater than 1.", BootstrapOutOfBag, 0
    )


def test_get_n_splits():
    oob = BootstrapOutOfBag(n_splits=3, random_seed=123)
    assert oob.n_splits == 3


@pytest.mark.parametrize("n_samples", [0, 1, 2])
def test_empty_test_indices_can_index_samples(n_samples):
    X = np.arange(n_samples).reshape(-1, 1)
    found_empty = False
    for train, test in BootstrapOutOfBag(n_splits=20, random_seed=0).split(X):
        assert X[train].shape == (n_samples, 1)
        assert X[test].shape == (len(test), 1)
        found_empty |= len(test) == 0
    assert found_empty


def test_test_indices_are_the_sorted_complement_of_training_samples():
    X = np.arange(1000).reshape(-1, 1)
    first = list(BootstrapOutOfBag(n_splits=5, random_seed=123).split(X))
    second = list(BootstrapOutOfBag(n_splits=5, random_seed=123).split(X))
    for (train, test), (repeated_train, repeated_test) in zip(first, second):
        assert_array_equal(test, np.setdiff1d(np.arange(len(X)), train))
        assert_array_equal(train, repeated_train)
        assert_array_equal(test, repeated_test)
        assert len(train) == len(X)
        assert len(np.unique(train)) < len(train)
