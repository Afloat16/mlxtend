# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mlxtend.evaluate import confusion_matrix


def test_multiclass():
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    x = np.array([[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    y = confusion_matrix(y_targ, y_pred, binary=False, positive_label=1)
    assert_array_equal(x, y)


def test_binary():
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    x = np.array([[4, 1], [1, 2]])
    y = confusion_matrix(y_targ, y_pred, binary=True, positive_label=1)
    assert_array_equal(x, y)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([-20, 5, 400]),
        np.array([2**63, 2**63 + 1, 2**64 - 1], dtype=np.uint64),
        np.array(["apple", "orange", "pear"]),
    ],
)
def test_repeated_label_pairs(labels):
    target = labels[[0, 0, 1, 2, 2, 2]]
    predicted = labels[[0, 1, 2, 0, 1, 2]]
    expected = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]])
    target = np.tile(target, 1000)
    predicted = np.tile(predicted, 1000)
    target_before = target.copy()
    predicted_before = predicted.copy()
    assert_array_equal(
        confusion_matrix(target, predicted),
        expected * 1000,
    )
    assert_array_equal(target, target_before)
    assert_array_equal(predicted, predicted_before)


def test_prediction_only_label_and_noncontiguous_inputs():
    target = np.array([8, 99, 8, 99, -3, 99])[::2]
    predicted = np.array([2, 99, 8, 99, -3, 99])[::2]
    assert_array_equal(
        confusion_matrix(target, predicted), [[1, 0, 0], [0, 0, 0], [0, 1, 1]]
    )
    assert_array_equal(target, [8, 8, -3])
    assert_array_equal(predicted, [2, 8, -3])


@pytest.mark.parametrize(
    "target, predicted, expected",
    [
        ([False, True, True], [False, False, True], [[1, 0], [1, 1]]),
        ([0, 0], [0, 0], [[2, 0], [0, 0]]),
        ([7, 7], [7, 7], [[0, 0], [0, 2]]),
        (["cat", "cat"], ["cat", "cat"], [[0, 0], [0, 2]]),
    ],
)
def test_boolean_and_single_class_counts(target, predicted, expected):
    assert_array_equal(confusion_matrix(target, predicted), expected)


def test_large_class_vocabulary():
    target = np.repeat(np.arange(100), 100)
    predicted = (target + 1) % 100
    expected = np.zeros((100, 100), dtype=int)
    expected[np.arange(100), (np.arange(100) + 1) % 100] = 100
    assert_array_equal(confusion_matrix(target, predicted), expected)


def test_string_binary_mapping():
    assert_array_equal(
        confusion_matrix(
            ["yes", "no", "maybe", "yes"],
            ["yes", "yes", "no", "no"],
            binary=True,
            positive_label="yes",
        ),
        [[1, 1], [1, 1]],
    )
