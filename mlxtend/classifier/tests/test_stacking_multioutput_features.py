import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from mlxtend.data import iris_data


def prediction_columns(model, samples, use_probas, drop):
    if not use_probas:
        prediction = model.predict(samples)
        return prediction.reshape(len(samples), -1)
    probabilities = model.predict_proba(samples)
    if not isinstance(probabilities, list):
        probabilities = [probabilities]
    columns = []
    for output in probabilities:
        first = 1 if drop == "first" else 0
        last = output.shape[1] - (drop == "last")
        columns.extend(output[:, column] for column in range(first, last))
    return np.column_stack(columns)


def exercise_stacking(
    kind, use_probas, drop, features, sparse_input, layout, average=False
):
    dense, labels = iris_data()
    if layout == "single":
        targets = labels.copy()
    elif layout == "binary":
        targets = np.column_stack([labels == 0, labels == 1]).astype(int)
    elif layout == "constant":
        targets = np.column_stack([np.zeros(len(labels), dtype=int), labels])
    else:
        targets = np.column_stack([labels % 2, labels])
    original_targets = targets.copy()
    samples = sparse.csr_matrix(dense) if sparse_input else dense.copy()
    base_models = [
        DecisionTreeClassifier(max_depth=2, random_state=11),
        DecisionTreeClassifier(max_depth=3, random_state=19),
    ]
    meta_model = DecisionTreeClassifier(max_depth=3, random_state=23)
    splits = list(KFold(n_splits=3, shuffle=True, random_state=7).split(dense))
    options = dict(
        classifiers=base_models,
        meta_classifier=meta_model,
        use_probas=use_probas,
        drop_proba_col=drop,
        use_features_in_secondary=features,
        store_train_meta_features=True,
    )
    if kind is StackingCVClassifier:
        options["cv"] = splits
    else:
        options["average_probas"] = average
    model = kind(**options)

    expected_train = []
    expected_test = []
    test_samples = samples[::7]
    for estimator in base_models:
        fitted = clone(estimator).fit(samples, targets)
        all_predictions = prediction_columns(fitted, dense, use_probas, drop)
        expected_test.append(prediction_columns(fitted, dense[::7], use_probas, drop))
        if kind is StackingCVClassifier:
            fold_predictions = np.empty_like(all_predictions)
            for train, test in splits:
                fold_model = clone(estimator).fit(samples[train], targets[train])
                fold_predictions[test] = prediction_columns(
                    fold_model, dense[test], use_probas, drop
                )
            expected_train.append(fold_predictions)
        else:
            expected_train.append(all_predictions)
    if average:
        train_features = sum(expected_train) / len(expected_train)
        test_features = sum(expected_test) / len(expected_test)
    else:
        train_features = np.column_stack(expected_train)
        test_features = np.column_stack(expected_test)

    model.fit(samples, targets)

    np.testing.assert_allclose(model.train_meta_features_, train_features)
    np.testing.assert_allclose(model.predict_meta_features(test_samples), test_features)
    if features:
        train_features = np.column_stack([dense, train_features])
        test_features = np.column_stack([dense[::7], test_features])
    reference = clone(meta_model).fit(train_features, targets)
    np.testing.assert_array_equal(
        model.predict(test_samples), reference.predict(test_features)
    )
    actual_proba = model.predict_proba(test_samples)
    expected_proba = reference.predict_proba(test_features)
    if layout == "single":
        np.testing.assert_allclose(actual_proba, expected_proba)
    else:
        assert len(actual_proba) == targets.shape[1]
        for actual, expected in zip(actual_proba, expected_proba):
            np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(targets, original_targets)


@pytest.mark.parametrize("kind", [StackingClassifier, StackingCVClassifier])
@pytest.mark.parametrize(
    "use_probas, drop", [(False, None), (True, None), (True, "first"), (True, "last")]
)
@pytest.mark.parametrize("features", [False, True])
@pytest.mark.parametrize("sparse_input", [False, True])
def test_multioutput_stacking_features(kind, use_probas, drop, features, sparse_input):
    exercise_stacking(kind, use_probas, drop, features, sparse_input, "unequal")


@pytest.mark.parametrize("kind", [StackingClassifier, StackingCVClassifier])
@pytest.mark.parametrize(
    "use_probas, drop", [(False, None), (True, None), (True, "first"), (True, "last")]
)
def test_single_output_stacking_controls(kind, use_probas, drop):
    exercise_stacking(kind, use_probas, drop, False, False, "single")


@pytest.mark.parametrize("drop", [None, "first", "last"])
def test_average_multilabel_probabilities(drop):
    exercise_stacking(
        StackingClassifier, True, drop, False, False, "binary", average=True
    )


@pytest.mark.parametrize("drop", [None, "first", "last"])
def test_constant_output_probability_columns(drop):
    exercise_stacking(StackingCVClassifier, True, drop, True, False, "constant")
