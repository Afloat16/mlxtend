import numpy as np
import pytest
from sklearn import config_context
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from mlxtend.feature_selection import (
    ExhaustiveFeatureSelector,
    SequentialFeatureSelector,
)


def make_selector(selector_class, estimator, scorer, cv, n_jobs):
    kwargs = {"scoring": scorer, "cv": cv, "n_jobs": n_jobs}
    if selector_class is SequentialFeatureSelector:
        return selector_class(estimator, k_features=2, **kwargs)
    return selector_class(
        estimator, min_features=2, max_features=2, print_progress=False, **kwargs
    )


def regression_data():
    rng = np.random.RandomState(37)
    samples = rng.normal(size=(24, 3))
    weights = np.tile([0.1, 1.0, 5.0, 20.0], 6)
    noise = rng.normal(size=24) * np.tile([1.0, 0.1, 2.0, 3.0], 6)
    targets = 2 * samples[:, 0] - 1.2 * samples[:, 1] + noise
    groups = np.repeat(np.arange(6), 4)
    return samples, targets, weights, groups


@pytest.mark.parametrize(
    "selector_class", [SequentialFeatureSelector, ExhaustiveFeatureSelector]
)
@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_routed_fit_and_score_match_independent_cv(
    selector_class, routing, grouped, n_jobs
):
    samples, targets, weights, groups = regression_data()
    cv = GroupKFold(3) if grouped else KFold(3)
    if not grouped:
        groups = None
    with config_context(enable_metadata_routing=True):
        estimator = Ridge(alpha=0.7).set_fit_request(sample_weight=True)
        scorer = make_scorer(
            mean_squared_error, greater_is_better=False
        ).set_score_request(sample_weight=True)

    with config_context(enable_metadata_routing=routing):
        selector = make_selector(selector_class, estimator, scorer, cv, n_jobs)
        selector.fit(samples, targets, groups=groups, sample_weight=weights)

        for record in selector.subsets_.values():
            kwargs = {"params": {"sample_weight": weights}}
            if routing and groups is not None:
                kwargs["params"]["groups"] = groups
            else:
                kwargs["groups"] = groups
            expected = cross_val_score(
                estimator,
                samples[:, record["feature_idx"]],
                targets,
                scoring=scorer,
                cv=cv,
                **kwargs,
            )
            np.testing.assert_allclose(record["cv_scores"], expected)
            np.testing.assert_allclose(record["avg_score"], expected.mean())


@pytest.mark.parametrize(
    "selector_class", [SequentialFeatureSelector, ExhaustiveFeatureSelector]
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_routing_supports_group_only_metadata(selector_class, n_jobs):
    samples, targets, _, groups = regression_data()
    cv = GroupKFold(3)
    with config_context(enable_metadata_routing=True):
        selector = make_selector(
            selector_class, Ridge(), "neg_mean_squared_error", cv, n_jobs
        ).fit(samples, targets, groups=groups)
        for record in selector.subsets_.values():
            expected = cross_val_score(
                Ridge(),
                samples[:, record["feature_idx"]],
                targets,
                scoring="neg_mean_squared_error",
                cv=cv,
                params={"groups": groups},
            )
            np.testing.assert_allclose(record["cv_scores"], expected)
