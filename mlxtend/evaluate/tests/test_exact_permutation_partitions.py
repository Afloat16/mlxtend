# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from mlxtend.evaluate import permutation_test


@pytest.mark.parametrize("m", [1, 2, 4, 5])
@pytest.mark.parametrize("direction", ["!=", ">", "<"])
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_exact_pvalue_matches_binary_assignment_oracle(m, direction, dtype):
    values = [7, -4, 2, 2, 0, 9]

    def statistic(a, b):
        difference = Fraction(sum(a), len(a)) - Fraction(sum(b), len(b))
        if direction == "!=":
            return abs(difference)
        return difference if direction == ">" else -difference

    reference = statistic(values[:m], values[m:])
    null_statistics = []
    for assignment in product([True, False], repeat=len(values)):
        if sum(assignment) == m:
            a = [v for v, selected in zip(values, assignment) if selected]
            b = [v for v, selected in zip(values, assignment) if not selected]
            null_statistics.append(statistic(a, b))
    expected = sum(v >= reference for v in null_statistics) / len(null_statistics)
    encoded = np.asarray(values, dtype=dtype)
    actual = permutation_test(
        encoded[:m], encoded[m:], func=f"x_mean {direction} y_mean", method="exact"
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, object])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_partitions_preserve_callback_order_dtype_and_inputs(dtype, reverse):
    values = [1, 4, 2, 4, 3, 1, 7]
    if dtype is object:
        values = [str(value) for value in values]
    backing = np.asarray(values, dtype=dtype)
    original = backing.copy()
    x, y = backing[::2], backing[1::2]
    if reverse:
        x, y = y, x
    combined = list(x) + list(y)
    expected_calls = []
    for assignment in product([True, False], repeat=len(combined)):
        if sum(assignment) == len(x):
            a = tuple(v for v, selected in zip(combined, assignment) if selected)
            b = tuple(v for v, selected in zip(combined, assignment) if not selected)
            expected_calls.append((a, b))
    calls = []
    marker = values[0]

    def statistic(a, b):
        assert a.dtype == backing.dtype
        assert b.dtype == backing.dtype
        calls.append((tuple(a), tuple(b)))
        return np.count_nonzero(a == marker) - np.count_nonzero(b == marker)

    actual = permutation_test(x, y, func=statistic)
    assert calls[0] == (tuple(x), tuple(y))
    assert calls[1:] == expected_calls
    reference = calls[0][0].count(marker) - calls[0][1].count(marker)
    expected = sum(
        a.count(marker) - b.count(marker) >= reference for a, b in expected_calls
    )
    assert actual == pytest.approx(expected / len(expected_calls))
    np.testing.assert_array_equal(backing, original)
