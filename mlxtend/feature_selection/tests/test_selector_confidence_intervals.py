import numpy as np
import pytest
from scipy.stats import sem, ttest_1samp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from mlxtend.feature_selection import (
    ExhaustiveFeatureSelector,
    SequentialFeatureSelector,
)


@pytest.mark.parametrize("selector_type", ["sequential", "exhaustive"])
@pytest.mark.parametrize("n_folds", [3, 5])
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_reported_bounds_match_student_mean_intervals(
    selector_type, n_folds, confidence
):
    rng = np.random.RandomState(17)
    samples = rng.normal(size=(60, 3))
    targets = 1.2 * samples[:, 0] + rng.normal(size=len(samples))
    splits = list(
        KFold(n_folds, shuffle=True, random_state=23).split(samples, targets)
    )
    options = dict(
        estimator=LinearRegression(),
        scoring="neg_mean_squared_error",
        cv=splits,
    )
    if selector_type == "sequential":
        selector = SequentialFeatureSelector(k_features=3, **options)
    else:
        selector = ExhaustiveFeatureSelector(
            min_features=1, max_features=3, print_progress=False, **options
        )
    selector.fit(samples, targets)

    metrics = selector.get_metric_dict(confidence_interval=confidence)

    for key, record in metrics.items():
        scores = cross_val_score(
            LinearRegression(),
            samples[:, record["feature_idx"]],
            targets,
            scoring="neg_mean_squared_error",
            cv=splits,
        )
        reference = ttest_1samp(scores, popmean=0).confidence_interval(confidence)
        mean = record["avg_score"]
        bound = record["ci_bound"]

        np.testing.assert_allclose(record["cv_scores"], scores, rtol=1e-12)
        np.testing.assert_allclose(
            [mean - bound, mean + bound], reference, rtol=1e-12
        )
        np.testing.assert_allclose(record["std_err"], sem(scores))
        np.testing.assert_allclose(record["std_dev"], np.std(scores))
        # get_metric_dict must not add reporting fields to fitted subsets.
        assert "ci_bound" not in selector.subsets_[key]
        assert "std_err" not in selector.subsets_[key]
