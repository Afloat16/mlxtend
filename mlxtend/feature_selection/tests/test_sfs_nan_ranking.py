import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from mlxtend.feature_selection import SequentialFeatureSelector

pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice")


def correlation(estimator, samples, targets):
    predicted = estimator.predict(samples)
    if np.ptp(predicted) == 0 or np.ptp(targets) == 0:
        return np.nan
    return pearsonr(targets, predicted).statistic


def data_and_cv(cv_kind):
    signal = np.linspace(-2, 3, 30)
    samples = np.column_stack((np.ones(30), signal, np.zeros(30), np.full(30, 2.0)))
    targets = 2 * signal
    if cv_kind == "partial":
        targets[:10] = 0
        cv = list(KFold(3).split(samples))
    elif cv_kind == "kfold":
        cv = list(KFold(3, shuffle=True, random_state=7).split(samples))
    else:
        cv = 0
    return samples, targets, cv


@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("cv_kind", ["none", "kfold", "partial"])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_nan_candidate_means_do_not_displace_scored_subsets(forward, cv_kind, n_jobs):
    samples, targets, cv = data_and_cv(cv_kind)
    model = SequentialFeatureSelector(
        LinearRegression(),
        k_features=3 if forward else 1,
        forward=forward,
        scoring=correlation,
        cv=cv,
        n_jobs=n_jobs,
    ).fit(samples, targets)

    for record in model.subsets_.values():
        indices = record["feature_idx"]
        assert 1 in indices
        if cv:
            expected = cross_val_score(
                LinearRegression(),
                samples[:, indices],
                targets,
                cv=cv,
                scoring=correlation,
            )
        else:
            fitted = LinearRegression().fit(samples[:, indices], targets)
            expected = [correlation(fitted, samples[:, indices], targets)]
        np.testing.assert_allclose(record["cv_scores"], expected, equal_nan=True)
        np.testing.assert_allclose(record["avg_score"], np.nanmean(expected))
        assert np.isfinite(record["avg_score"])
    assert np.isfinite(model.k_score_)


@pytest.mark.parametrize("forward", [False, True])
def test_nan_ranking_respects_fixed_features(forward):
    samples, targets, cv = data_and_cv("kfold")
    model = SequentialFeatureSelector(
        LinearRegression(),
        k_features=2,
        forward=forward,
        fixed_features=[0],
        scoring=correlation,
        cv=cv,
    ).fit(samples, targets)

    assert model.k_feature_idx_ == (0, 1)
    np.testing.assert_allclose(model.k_score_, 1.0)


@pytest.mark.parametrize("forward", [False, True])
def test_nan_ranking_respects_feature_groups(forward):
    samples, targets, cv = data_and_cv("partial")
    model = SequentialFeatureSelector(
        LinearRegression(),
        k_features=1,
        forward=forward,
        feature_groups=[[0, 2, 3], [1]],
        scoring=correlation,
        cv=cv,
    ).fit(samples, targets)

    assert model.k_feature_idx_ == (1,)
    np.testing.assert_allclose(model.k_score_, 1.0)
    assert np.isnan(model.subsets_[1]["cv_scores"][0])


@pytest.mark.parametrize("cv_kind", ["none", "kfold"])
def test_all_nan_candidates_retain_existing_first_candidate_behavior(cv_kind):
    samples, targets, cv = data_and_cv(cv_kind)
    targets[:] = 1
    model = SequentialFeatureSelector(
        LinearRegression(), k_features=1, scoring=correlation, cv=cv
    ).fit(samples, targets)

    assert model.k_feature_idx_ == (0,)
    assert np.isnan(model.k_score_)
