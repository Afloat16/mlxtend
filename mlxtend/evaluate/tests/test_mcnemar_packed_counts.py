# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

from itertools import combinations

import numpy as np
import pytest
from scipy.stats import binomtest

from mlxtend.evaluate import mcnemar, mcnemar_tables


def _count_table(target, first, second):
    # Independent per-observation contingency-table definition.
    counts = [[0, 0], [0, 0]]
    for truth, p1, p2 in zip(target, first, second):
        counts[int(p1 != truth)][int(p2 != truth)] += 1
    return np.array(counts, dtype=float)


@pytest.mark.parametrize("n_samples", [0, 1, 7, 8, 9, 15, 16, 17, 257, 2057])
@pytest.mark.parametrize("n_models", [2, 5])
def test_tables_match_direct_counts_at_byte_boundaries(n_samples, n_models):
    rng = np.random.RandomState(5)
    target = rng.randint(0, 4, size=n_samples)
    predictions = [rng.randint(0, 4, size=n_samples) for _ in range(n_models)]
    # Also exercise perfect and completely incorrect models.
    predictions[0] = target.copy()
    predictions[1] = (target + 1) % 4
    target_before = target.copy()
    predictions_before = [values.copy() for values in predictions]
    result = mcnemar_tables(target, *predictions)
    expected_keys = [
        f"model_{i} vs model_{j}" for i, j in combinations(range(n_models), 2)
    ]
    assert list(result) == expected_keys
    for i, j in combinations(range(n_models), 2):
        table = result[f"model_{i} vs model_{j}"]
        np.testing.assert_array_equal(
            table, _count_table(target, predictions[i], predictions[j])
        )
        assert table.dtype == np.dtype(float)
        assert table.sum() == n_samples
    np.testing.assert_array_equal(target, target_before)
    for actual, before in zip(predictions, predictions_before):
        np.testing.assert_array_equal(actual, before)


@pytest.mark.parametrize("dtype", [str, bytes, object])
def test_label_dtypes_and_noncontiguous_inputs(dtype):
    target = np.tile(np.array(["cat", "dog", "bird", "cat"], dtype=dtype), 131)
    predictions = [target.copy(), np.roll(target, 1), np.roll(target, 2)]
    target = target[::2]
    predictions = [pred[::2] for pred in predictions]
    target.flags.writeable = False
    for pred in predictions:
        pred.flags.writeable = False
    result = mcnemar_tables(target, *predictions)
    for i, j in combinations(range(3), 2):
        np.testing.assert_array_equal(
            result[f"model_{i} vs model_{j}"],
            _count_table(target, predictions[i], predictions[j]),
        )


@pytest.mark.parametrize("dtype", [float, object])
def test_nan_labels_remain_incorrect_even_when_both_predictions_are_nan(dtype):
    target = np.tile(np.array([np.nan, 1, 0, 1, np.nan], dtype=dtype), 61)
    predictions = [target.copy(), np.roll(target, 1), np.zeros_like(target)]
    result = mcnemar_tables(target, *predictions)
    for i, j in combinations(range(3), 2):
        np.testing.assert_array_equal(
            result[f"model_{i} vs model_{j}"],
            _count_table(target, predictions[i], predictions[j]),
        )


def test_model_order_swaps_table_axes_and_preserves_exact_pvalue():
    target = np.arange(519) % 3
    predictions = [target.copy() for _ in range(4)]
    for index, prediction in enumerate(predictions):
        prediction[index :: index + 2] = (prediction[index :: index + 2] + 1) % 3
    forward = mcnemar_tables(target, *predictions)
    reverse = mcnemar_tables(target, *predictions[::-1])
    for i, j in combinations(range(4), 2):
        table = forward[f"model_{i} vs model_{j}"]
        reversed_table = reverse[f"model_{3 - j} vs model_{3 - i}"]
        np.testing.assert_array_equal(table, reversed_table.T)
        b, c = int(table[0, 1]), int(table[1, 0])
        expected_p = binomtest(b, n=b + c).pvalue if b + c else 1.0
        statistic, pvalue = mcnemar(table, exact=True)
        assert statistic is None
        np.testing.assert_allclose(pvalue, expected_p, rtol=1e-12)


def test_identical_models_and_one_disagreement_after_full_bytes():
    target = np.ones(1033, dtype=int)
    first = target.copy()
    second = target.copy()
    second[-1] = 0
    result = mcnemar_tables(target, first, first.copy(), second)
    np.testing.assert_array_equal(result["model_0 vs model_1"], [[1033, 0], [0, 0]])
    np.testing.assert_array_equal(result["model_0 vs model_2"], [[1032, 1], [0, 0]])
