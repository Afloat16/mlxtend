"""Bootstrap replicates must not reuse fitted warm-start state when cloning."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from mlxtend.evaluate import bootstrap_point632_score


def regression_data():
    values = np.arange(24, dtype=float)
    return np.column_stack([values, values % 5]), values**2 / 10 + np.sin(values)


def independent_scores(estimator, x, y, seed, method, fresh=True, probabilities=False):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(y))
    model = clone(estimator)
    scores = []
    for _ in range(4):
        train = rng.choice(indices, size=len(y), replace=True)
        test = np.flatnonzero(~np.isin(indices, train))
        assert len(test)
        if fresh:
            model = clone(estimator)
        model.fit(x[train], y[train])
        predict = (
            (lambda data: model.predict_proba(data)[:, 1])
            if probabilities
            else model.predict
        )
        test_loss = np.mean((y[test] - predict(x[test])) ** 2)
        if method == "oob":
            scores.append(test_loss)
        else:
            training_loss = np.mean((y - predict(x)) ** 2)
            scores.append(0.632 * test_loss + 0.368 * training_loss)
    return np.array(scores)


@pytest.mark.parametrize("seed", [1, 7])
@pytest.mark.parametrize("method", ["oob", ".632"])
@pytest.mark.parametrize(
    "estimator_class", [RandomForestRegressor, GradientBoostingRegressor]
)
def test_warm_start_replicates_match_fresh_standalone_fits(
    estimator_class, method, seed
):
    x, y = regression_data()
    estimator = estimator_class(
        n_estimators=3, max_depth=2, warm_start=True, random_state=13
    )
    expected = independent_scores(estimator, x, y, seed, method)
    actual = bootstrap_point632_score(
        estimator, x, y, n_splits=4, method=method, random_seed=seed
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert not hasattr(estimator, "estimators_")


@pytest.mark.parametrize("seed", [1, 7])
def test_probability_predictions_use_each_new_clone(seed):
    x, _ = regression_data()
    y = (np.arange(len(x)) % 3 == 0).astype(int)
    estimator = RandomForestClassifier(
        n_estimators=3, max_depth=2, warm_start=True, random_state=13
    )
    expected = independent_scores(estimator, x, y, seed, ".632", probabilities=True)
    actual = bootstrap_point632_score(
        estimator,
        x,
        y,
        n_splits=4,
        method=".632",
        predict_proba=True,
        scoring_func=lambda actual, predicted: np.mean((actual - predicted) ** 2),
        random_seed=seed,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert not hasattr(estimator, "estimators_")


@pytest.mark.parametrize("method", ["oob", ".632"])
def test_cold_start_estimators_keep_their_scores(method):
    x, y = regression_data()
    estimator = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=13)
    expected = independent_scores(estimator, x, y, 7, method)
    actual = bootstrap_point632_score(
        estimator, x, y, n_splits=4, method=method, random_seed=7
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_disabling_cloning_preserves_explicit_estimator_reuse():
    x, y = regression_data()
    estimator = RandomForestRegressor(
        n_estimators=3, max_depth=2, warm_start=True, random_state=13
    )
    expected = independent_scores(estimator, x, y, 7, "oob", fresh=False)
    actual = bootstrap_point632_score(
        estimator, x, y, n_splits=4, method="oob", random_seed=7, clone_estimator=False
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    assert len(estimator.estimators_) == 3


def test_previously_fitted_input_model_is_preserved():
    x, y = regression_data()
    estimator = RandomForestRegressor(
        n_estimators=3, max_depth=2, warm_start=True, random_state=13
    ).fit(x, y)
    original_predictions = estimator.predict(x)
    original_trees = tuple(estimator.estimators_)
    expected = independent_scores(estimator, x, y, 7, "oob")
    actual = bootstrap_point632_score(
        estimator, x, y, n_splits=4, method="oob", random_seed=7
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    np.testing.assert_array_equal(estimator.predict(x), original_predictions)
    assert tuple(estimator.estimators_) == original_trees
