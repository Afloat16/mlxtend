import numpy as np
import pytest
from scipy.stats import sem
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from mlxtend.feature_selection import SequentialFeatureSelector


def data(seed):
    rng = np.random.RandomState(seed)
    samples = rng.normal(size=(45, 5))
    targets = 1.4 * samples[:, 0] + 0.3 * samples[:, 1]
    targets += rng.normal(scale=1.2, size=len(samples))
    return samples, targets


def reference_choice(selector, samples, targets, splits):
    scores = {}
    for size, record in selector.subsets_.items():
        scores[size] = cross_val_score(
            LinearRegression(),
            samples[:, record["feature_idx"]],
            targets,
            cv=splits,
            scoring="neg_mean_squared_error",
        )
        np.testing.assert_allclose(scores[size], record["cv_scores"], rtol=1e-12)
    best = max(scores, key=lambda size: scores[size].mean())
    threshold = scores[best].mean() - sem(scores[best])
    eligible = [size for size in scores if scores[size].mean() >= threshold]
    chosen = min(eligible)
    return chosen, scores[chosen].mean(), threshold


@pytest.mark.parametrize("seed", [3, 7, 13])
@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("floating", [False, True])
def test_parsimonious_uses_best_subset_standard_error(seed, forward, floating):
    samples, targets = data(seed)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=11).split(samples))
    selector = SequentialFeatureSelector(
        LinearRegression(),
        k_features="parsimonious",
        forward=forward,
        floating=floating,
        scoring="neg_mean_squared_error",
        cv=splits,
    ).fit(samples, targets)
    chosen, score, threshold = reference_choice(selector, samples, targets, splits)

    assert selector.k_feature_idx_ == selector.subsets_[chosen]["feature_idx"]
    np.testing.assert_allclose(selector.k_score_, score)
    assert selector.k_score_ >= threshold
    for size in range(1, chosen):
        assert selector.subsets_[size]["avg_score"] < threshold


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("floating", [False, True])
def test_parsimonious_respects_fixed_feature_groups(forward, floating):
    samples, targets = data(7)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=11).split(samples))
    selector = SequentialFeatureSelector(
        LinearRegression(),
        k_features="parsimonious",
        forward=forward,
        floating=floating,
        scoring="neg_mean_squared_error",
        cv=splits,
        feature_groups=[[0, 1], [2], [3], [4]],
        fixed_features=[2],
    ).fit(samples, targets)
    chosen, score, _ = reference_choice(selector, samples, targets, splits)

    assert selector.k_feature_idx_ == selector.subsets_[chosen]["feature_idx"]
    assert 2 in selector.k_feature_idx_
    assert (0 in selector.k_feature_idx_) == (1 in selector.k_feature_idx_)
    np.testing.assert_allclose(selector.k_score_, score)


@pytest.mark.parametrize("forward", [False, True])
def test_parsimonious_without_cv_uses_zero_standard_error(forward):
    samples, targets = data(3)
    selector = SequentialFeatureSelector(
        LinearRegression(),
        k_features="parsimonious",
        forward=forward,
        scoring="neg_mean_squared_error",
        cv=0,
    ).fit(samples, targets)
    best_score = max(record["avg_score"] for record in selector.subsets_.values())
    expected = min(
        size
        for size, record in selector.subsets_.items()
        if record["avg_score"] == best_score
    )

    assert selector.k_feature_idx_ == selector.subsets_[expected]["feature_idx"]
    np.testing.assert_allclose(selector.k_score_, best_score)
