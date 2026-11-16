# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlxtend.classifier import StackingCVClassifier
from mlxtend.regressor import StackingCVRegressor


@pytest.mark.parametrize("kind", ["classifier", "probabilities", "regressor"])
@pytest.mark.parametrize("routing", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("passthrough", [False, True])
def test_grouped_stacking_matches_independent_fold_fits(
    kind, routing, weighted, passthrough
):
    rng = np.random.RandomState(4)
    X = rng.normal(size=(48, 3))
    groups = np.repeat([12, 8, 41, 3, 19, 7], 8)
    weights = np.linspace(0.1, 4, len(X)) if weighted else None
    if kind == "regressor":
        y = 2 * X[:, 0] - X[:, 1] + rng.normal(size=len(X))
        bases = [DecisionTreeRegressor(max_depth=d, random_state=2) for d in (1, 3)]
        meta = Ridge(alpha=0.7)
        stack_type = StackingCVRegressor
        extra = {}
    else:
        y = np.tile([0, 1], 24)
        bases = [DecisionTreeClassifier(max_depth=d, random_state=2) for d in (1, 3)]
        meta = LogisticRegression(C=0.7, max_iter=500)
        stack_type = StackingCVClassifier
        extra = {"use_probas": kind == "probabilities"}

    originals = [a.copy() for a in (X, y, groups)]
    original_weights = None if weights is None else weights.copy()
    with config_context(enable_metadata_routing=routing):
        if routing and weighted:
            bases = [model.set_fit_request(sample_weight=True) for model in bases]
        model = stack_type(
            bases,
            meta,
            cv=GroupKFold(3),
            store_train_meta_features=True,
            use_features_in_secondary=passthrough,
            **extra,
        ).fit(X, y, groups=groups, sample_weight=weights)

        # Construct out-of-fold features explicitly. Every holdout group must
        # be absent from the training fold, with weights selected by row.
        expected_columns = []
        full_models = []
        for base in bases:
            width = 2 if kind == "probabilities" else 1
            predictions = np.empty((len(X), width))
            for train, test in GroupKFold(3).split(X, y, groups):
                assert set(groups[train]).isdisjoint(groups[test])
                fit_kwargs = (
                    {} if weights is None else {"sample_weight": weights[train]}
                )
                fitted = clone(base).fit(X[train], y[train], **fit_kwargs)
                if kind == "probabilities":
                    predictions[test] = fitted.predict_proba(X[test])
                else:
                    predictions[test, 0] = fitted.predict(X[test])
            expected_columns.append(predictions)
            fit_kwargs = {} if weights is None else {"sample_weight": weights}
            full_models.append(clone(base).fit(X, y, **fit_kwargs))
        expected_oof = np.column_stack(expected_columns)
        np.testing.assert_allclose(model.train_meta_features_, expected_oof)

        train_meta = np.column_stack((X, expected_oof)) if passthrough else expected_oof
        expected_meta = clone(meta).fit(train_meta, y, **fit_kwargs)
        X_new = rng.normal(size=(11, X.shape[1]))
        prediction_columns = [
            (
                base.predict_proba(X_new)
                if kind == "probabilities"
                else base.predict(X_new)
            )
            for base in full_models
        ]
        test_meta = np.column_stack(prediction_columns)
        np.testing.assert_allclose(model.predict_meta_features(X_new), test_meta)
        if passthrough:
            test_meta = np.column_stack((X_new, test_meta))
        np.testing.assert_allclose(
            model.predict(X_new), expected_meta.predict(test_meta)
        )
        if kind != "regressor":
            np.testing.assert_allclose(
                model.predict_proba(X_new), expected_meta.predict_proba(test_meta)
            )

    for actual, original in zip((X, y, groups), originals):
        np.testing.assert_array_equal(actual, original)
    if weights is not None:
        np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize("kind", ["classifier", "regressor"])
@pytest.mark.parametrize("weighted", [False, True])
def test_metadata_routing_without_groups(kind, weighted):
    X = np.arange(48, dtype=float).reshape(24, 2)
    y = np.tile([0, 1], 12) if kind == "classifier" else np.arange(24, dtype=float)
    weights = np.linspace(0.1, 2, len(X)) if weighted else None
    stack_type = StackingCVClassifier if kind == "classifier" else StackingCVRegressor
    base = (
        DecisionTreeClassifier(max_depth=2, random_state=2)
        if kind == "classifier"
        else Ridge()
    )
    meta = LogisticRegression() if kind == "classifier" else Ridge()

    with config_context(enable_metadata_routing=False):
        reference = stack_type([clone(base)], clone(meta), cv=KFold(3)).fit(
            X, y, sample_weight=weights
        )
    with config_context(enable_metadata_routing=True):
        if weighted:
            base.set_fit_request(sample_weight=True)
        actual = stack_type([base], meta, cv=KFold(3)).fit(X, y, sample_weight=weights)
        np.testing.assert_allclose(actual.predict(X), reference.predict(X))
