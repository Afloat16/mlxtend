"""R-squared permutation scores must not inherit narrow target arithmetic."""

from fractions import Fraction

import numpy as np
import pytest

from mlxtend.evaluate import feature_importance_permutation


@pytest.mark.parametrize("feature_groups", [None, [0]])
@pytest.mark.parametrize(
    "values,dtype",
    [
        ([0, 10000, 20000, 30000], np.int16),
        ([0, 10000, 20000, 30000], np.uint16),
        ([0, 50000, 100000, 150000], np.int32),
        ([0, 10**12, 2 * 10**12, 3 * 10**12], np.int64),
        ([0, 2**62, 2**63, 3 * 2**62], np.uint64),
        ([0, 300, 600, 900], np.float16),
        ([0, 1e20, 2e20, 3e20], np.float32),
        ([0, 0.5, 1, 1.5], np.float64),
    ],
)
def test_r2_importance_matches_exact_scalar_arithmetic(values, dtype, feature_groups):
    y = np.array(values, dtype=dtype)
    x = y[:, None].copy()
    original_x, original_y = x.copy(), y.copy()
    mean_importance, all_importance = feature_importance_permutation(
        x,
        y,
        predict_method=lambda data: data[:, 0],
        metric="r2",
        num_rounds=3,
        feature_groups=feature_groups,
        seed=7,
    )

    # Fractions give an independent exact sum-of-squares reference, using
    # the values actually represented by the input dtype.
    exact = [
        (
            Fraction(int(value))
            if np.issubdtype(dtype, np.integer)
            else Fraction(float(value))
        )
        for value in y
    ]
    mean = sum(exact) / len(exact)
    denominator = sum((value - mean) ** 2 for value in exact)
    rng = np.random.RandomState(7)
    expected = []
    for _ in range(3):
        permutation = rng.permutation(len(y))
        numerator = sum(
            (value - exact[index]) ** 2 for value, index in zip(exact, permutation)
        )
        expected.append(float(numerator / denominator))

    np.testing.assert_allclose(all_importance[0], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mean_importance, [np.mean(expected)], rtol=1e-12)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)


def test_custom_scoring_retains_original_target_and_prediction_dtypes():
    y = np.array([0, 1, 2, 3], dtype=np.int16)
    seen = []

    def scorer(actual, predicted):
        seen.append((actual.dtype, predicted.dtype))
        return float(np.mean(actual == predicted))

    feature_importance_permutation(
        y[:, None].copy(), y, lambda data: data[:, 0], metric=scorer, seed=7
    )
    assert seen == [(np.dtype("int16"), np.dtype("int16"))] * 2
