# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import StackingClassifier, StackingCVClassifier


class _ObjectPredictionsTree(DecisionTreeClassifier):
    def predict(self, X, check_input=True):
        return super().predict(X, check_input=check_input).astype(object)


def _make_stack(stack_type, **kwargs):
    cv_params = {}
    if stack_type is StackingCVClassifier:
        cv_params["cv"] = StratifiedKFold(3, shuffle=True, random_state=17)
    return stack_type(
        classifiers=[DecisionTreeClassifier(max_depth=2, random_state=1), GaussianNB()],
        meta_classifier=LogisticRegression(random_state=2),
        store_train_meta_features=True,
        **cv_params,
        **kwargs,
    )


def _training_predictions(stack, X, y, method="predict"):
    if isinstance(stack, StackingCVClassifier):
        return [
            cross_val_predict(clf, X, y, cv=stack.cv, method=method)
            for clf in stack.classifiers
        ]
    return [getattr(clf, method)(X) for clf in stack.clfs_]


@pytest.mark.parametrize("stack_type", [StackingClassifier, StackingCVClassifier])
@pytest.mark.parametrize("dtype", [str, object])
@pytest.mark.parametrize("include_features", [False, True])
def test_string_targets_use_consistent_numeric_meta_features(
    stack_type, dtype, include_features
):
    X, target = make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=0, random_state=11
    )
    y = np.array(["cat", "dog"], dtype=dtype)[target]
    original_y = y.copy()
    stack = _make_stack(stack_type, use_features_in_secondary=include_features).fit(
        X, y
    )

    def encode(predictions):
        return np.column_stack(
            [[{"cat": 0, "dog": 1}[label] for label in pred] for pred in predictions]
        )

    expected_train = encode(_training_predictions(stack, X, y))
    np.testing.assert_array_equal(stack.train_meta_features_, expected_train)
    meta_X = (
        np.column_stack((X, expected_train)) if include_features else expected_train
    )
    independent_meta = clone(stack.meta_classifier).fit(meta_X, y)

    test_X = X[:9] + 0.25
    expected_test = encode([clf.predict(test_X) for clf in stack.clfs_])
    np.testing.assert_array_equal(stack.predict_meta_features(test_X), expected_test)
    meta_test = (
        np.column_stack((test_X, expected_test)) if include_features else expected_test
    )
    np.testing.assert_array_equal(
        stack.predict(test_X), independent_meta.predict(meta_test)
    )
    np.testing.assert_allclose(
        stack.predict_proba(test_X), independent_meta.predict_proba(meta_test)
    )
    np.testing.assert_allclose(
        stack.decision_function(test_X), independent_meta.decision_function(meta_test)
    )
    np.testing.assert_array_equal(stack.classes_, ["cat", "dog"])
    np.testing.assert_array_equal(y, original_y)


@pytest.mark.parametrize("stack_type", [StackingClassifier, StackingCVClassifier])
def test_numeric_meta_labels_keep_their_values(stack_type):
    X = np.arange(60).reshape(30, 2)
    y = np.array([-3, 7] * 15)
    stack = _make_stack(stack_type).fit(X, y)
    np.testing.assert_array_equal(
        stack.train_meta_features_, np.column_stack(_training_predictions(stack, X, y))
    )
    np.testing.assert_array_equal(
        stack.predict_meta_features(X),
        np.column_stack([clf.predict(X) for clf in stack.clfs_]),
    )


@pytest.mark.parametrize("stack_type", [StackingClassifier, StackingCVClassifier])
def test_numeric_labels_returned_as_object_keep_their_values(stack_type):
    X = np.arange(60).reshape(30, 2)
    y = np.array([-3, 7] * 15)
    stack = _make_stack(stack_type)
    stack.classifiers = [_ObjectPredictionsTree(max_depth=2, random_state=1)]
    stack.fit(X, y)
    np.testing.assert_array_equal(
        stack.train_meta_features_, np.column_stack(_training_predictions(stack, X, y))
    )
    np.testing.assert_array_equal(
        stack.predict_meta_features(X), stack.clfs_[0].predict(X)[:, np.newaxis]
    )


@pytest.mark.parametrize("stack_type", [StackingClassifier, StackingCVClassifier])
def test_string_targets_leave_probability_features_unchanged(stack_type):
    X = np.arange(60).reshape(30, 2)
    y = np.array(["cat", "dog"] * 15)
    stack = _make_stack(stack_type, use_probas=True).fit(X, y)
    np.testing.assert_allclose(
        stack.train_meta_features_,
        np.column_stack(_training_predictions(stack, X, y, method="predict_proba")),
    )
    np.testing.assert_allclose(
        stack.predict_meta_features(X),
        np.column_stack([clf.predict_proba(X) for clf in stack.clfs_]),
    )


@pytest.mark.parametrize("stack_type", [StackingClassifier, StackingCVClassifier])
def test_string_target_stacking_inside_cross_validation(stack_type):
    X = np.arange(120).reshape(60, 2)
    y = np.array(["cat", "dog"] * 30)
    scores = cross_val_score(_make_stack(stack_type), X, y, cv=3, error_score="raise")
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_prefitted_string_classifiers_are_not_refitted():
    X = np.arange(60).reshape(30, 2)
    y = np.array(["cat", "dog"] * 15)
    base = DecisionTreeClassifier(random_state=1).fit(X, y)
    tree = base.tree_
    stack = StackingClassifier(
        classifiers=[base],
        meta_classifier=LogisticRegression(),
        fit_base_estimators=False,
    ).fit(X, y)
    assert stack.clfs_[0] is base
    assert base.tree_ is tree
    np.testing.assert_array_equal(stack.predict(X), y)


@pytest.mark.parametrize("stack_type", [StackingClassifier, StackingCVClassifier])
def test_numeric_looking_strings_use_class_vocabulary_order(stack_type):
    X = np.arange(60).reshape(30, 2)
    y = np.array(["10", "2"] * 15)
    stack = _make_stack(stack_type).fit(X, y)
    expected = np.column_stack(
        [[{"10": 0, "2": 1}[label] for label in clf.predict(X)] for clf in stack.clfs_]
    )
    np.testing.assert_array_equal(stack.predict_meta_features(X), expected)


def test_stacking_classifier_encodes_each_string_output_separately():
    X = np.arange(60).reshape(30, 2)
    y = np.column_stack(
        [np.array(["cat", "dog"] * 15), np.array(["green", "pink", "pink"] * 10)]
    )
    stack = StackingClassifier(
        classifiers=[DecisionTreeClassifier(random_state=1)],
        meta_classifier=MultiOutputClassifier(LogisticRegression()),
        store_train_meta_features=True,
    ).fit(X, y)
    expected = np.column_stack([y[:, 0] == "dog", y[:, 1] == "pink"]).astype(int)
    np.testing.assert_array_equal(stack.train_meta_features_, expected)
    np.testing.assert_array_equal(stack.predict_meta_features(X), expected)
    np.testing.assert_array_equal(stack.predict(X), y)
