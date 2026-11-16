# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_predict

from mlxtend.classifier import MultiLayerPerceptron, SoftmaxRegression


@pytest.fixture(params=[SoftmaxRegression, MultiLayerPerceptron])
def model(request):
    kwargs = {"hidden_layers": [3]} if request.param is MultiLayerPerceptron else {}
    return request.param(epochs=2, eta=0.01, random_seed=0, **kwargs)


@pytest.mark.parametrize("labels", [[0, 2], [1, 2], [0, 1], [0, 1, 2]])
@pytest.mark.parametrize("n_classes", [None, 3])
def test_probability_columns_name_every_fitted_output(model, labels, n_classes):
    y = np.tile(labels, 3)
    X = np.random.RandomState(13).normal(size=(len(y), 2))
    before = X.copy(), y.copy()
    model.set_params(n_classes=n_classes).fit(X, y)
    expected = np.arange(max(labels) + 1 if n_classes is None else n_classes)
    np.testing.assert_array_equal(model.classes_, expected)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), len(expected))
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    np.testing.assert_array_equal(
        model.predict(X), model.classes_[proba.argmax(axis=1)]
    )
    np.testing.assert_array_equal(X, before[0])
    np.testing.assert_array_equal(y, before[1])


def test_declared_outputs_include_unobserved_end_classes(model):
    X = np.arange(16, dtype=float).reshape(8, 2) / 16
    y = np.tile([1, 3], 4)
    model.set_params(n_classes=5).fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.arange(5))
    assert model.predict_proba(X).shape == (8, 5)


def test_continued_fit_keeps_metadata_for_all_existing_outputs(model):
    X = np.random.RandomState(3).normal(size=(12, 2))
    y = np.tile(np.arange(3), 4)
    model.fit(X, y)
    subset = y != 1
    model.fit(X[subset], y[subset], init_params=False)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    assert model.predict_proba(X).shape == (12, 3)


def test_cross_val_predict_with_missing_training_classes_matches_manual_folds(model):
    X = np.random.RandomState(2).normal(size=(12, 2))
    y = np.tile(np.arange(3), 4)
    folds = [
        (np.flatnonzero(y != label), np.flatnonzero(y == label)) for label in range(3)
    ]
    model.set_params(n_classes=3)
    expected = np.empty((len(y), 3))
    for train, test in folds:
        fitted = clone(model).fit(X[train], y[train])
        expected[test] = fitted.predict_proba(X[test])
    actual = cross_val_predict(model, X, y, cv=folds, method="predict_proba")
    np.testing.assert_array_equal(actual, expected)


def test_probability_scorer_uses_the_full_declared_label_space(model):
    X = np.random.RandomState(7).normal(size=(12, 2))
    y = np.tile(np.arange(3), 4)
    subset = y != 1
    model.set_params(n_classes=3).fit(X[subset], y[subset])
    proba = model.predict_proba(X)
    expected = np.log(proba[np.arange(len(y)), y]).mean()
    actual = get_scorer("neg_log_loss")(model, X, y)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
