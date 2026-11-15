import numpy as np
import pytest
from scipy import stats

from mlxtend.evaluate import ftest


CORRECT = np.array(
    [
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 1],
    ],
    dtype=float,
)


def linear_model_reference(correct):
    n_samples, n_models = correct.shape
    response = correct.ravel()
    # Restricted model: one intercept per sample. Full model: also include
    # model effects. Residual degrees of freedom follow the design rank.
    restricted = np.repeat(np.eye(n_samples), n_models, axis=0)
    model_effects = np.tile(np.eye(n_models)[:, 1:], (n_samples, 1))
    full = np.column_stack([restricted, model_effects])
    residual_sums = []
    ranks = []
    for design in [restricted, full]:
        coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
        residual_sums.append(np.sum((response - design @ coefficients) ** 2))
        ranks.append(rank)
    model_df = ranks[1] - ranks[0]
    residual_df = response.size - ranks[1]
    statistic = ((residual_sums[0] - residual_sums[1]) / model_df) / (
        residual_sums[1] / residual_df
    )
    return statistic, stats.f.sf(statistic, model_df, residual_df)


@pytest.mark.parametrize("n_samples", [4, 6, 12])
@pytest.mark.parametrize("n_models", [2, 3, 4])
def test_ftest_matches_repeated_measures_linear_model(n_samples, n_models):
    correct = np.tile(CORRECT, (2, 1))[:n_samples, :n_models]
    target = np.arange(n_samples) % 2
    predictions = np.where(correct, target[:, None], 1 - target[:, None])
    original_predictions = predictions.copy()
    expected = linear_model_reference(correct)

    actual = ftest(target, *predictions.T)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(predictions, original_predictions)


@pytest.mark.parametrize(
    "correct",
    [
        [[1, 0], [1, 0], [1, 0], [0, 0]],
        [[1, 1], [1, 0], [1, 0], [0, 1], [1, 0], [0, 0]],
        [[0, 1], [0, 1], [0, 0], [1, 1], [1, 1]],
    ],
)
def test_two_model_ftest_agrees_with_paired_ttest(correct):
    correct = np.asarray(correct, dtype=float)
    expected = stats.ttest_rel(correct[:, 0], correct[:, 1])

    statistic, p_value = ftest(np.ones(len(correct), dtype=int), *correct.T)

    np.testing.assert_allclose(statistic, expected.statistic**2, rtol=1e-12)
    np.testing.assert_allclose(p_value, expected.pvalue, rtol=1e-12)


def test_equal_accuracies_keep_unit_p_value():
    target = np.ones(4, dtype=int)
    first = np.array([1, 0, 1, 0])
    second = np.array([0, 1, 0, 1])

    np.testing.assert_allclose(ftest(target, first, second), [0.0, 1.0])
