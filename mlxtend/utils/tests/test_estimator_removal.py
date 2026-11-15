import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from mlxtend.regressor import StackingCVRegressor, StackingRegressor

KINDS = ["classifier", "cv_classifier", "regressor", "cv_regressor"]


def components(kind):
    if "classifier" in kind:
        return [
            DecisionTreeClassifier(max_depth=2, random_state=11),
            GaussianNB(),
            KNeighborsClassifier(n_neighbors=3),
        ]
    return [DummyRegressor(), Ridge(alpha=0.7), SVR(C=0.8)]


def make_stack(kind, estimators):
    classes = {
        "classifier": StackingClassifier,
        "cv_classifier": StackingCVClassifier,
        "regressor": StackingRegressor,
        "cv_regressor": StackingCVRegressor,
    }
    kwargs = {"cv": 3, "shuffle": False} if kind.startswith("cv_") else {}
    if "classifier" in kind:
        return classes[kind](
            classifiers=estimators,
            meta_classifier=LogisticRegression(max_iter=200, random_state=7),
            **kwargs,
        )
    return classes[kind](
        regressors=estimators, meta_regressor=Ridge(alpha=0.4), **kwargs
    )


def data(kind):
    rng = np.random.RandomState(21)
    samples = rng.normal(size=(60, 3))
    targets = 1.8 * samples[:, 0] - samples[:, 1] + rng.normal(scale=0.3, size=60)
    if "classifier" in kind:
        targets = (targets > np.median(targets)).astype(int)
    return samples, targets


def root_and_names(stack, kind):
    if "classifier" in kind:
        return "classifiers", [name for name, _ in stack.named_classifiers]
    return "regressors", [name for name, _ in stack.named_regressors]


def assert_fitted_equivalent(actual, expected, kind):
    samples, targets = data(kind)
    actual.fit(samples, targets)
    expected.fit(samples, targets)
    points = samples[::-1] + 0.3
    np.testing.assert_allclose(actual.predict(points), expected.predict(points))
    root, _ = root_and_names(actual, kind)
    assert [type(item) for item in getattr(actual, root)] == [
        type(item) for item in getattr(expected, root)
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("action", ["remove_last", "replace_middle", "replace_last"])
def test_named_removal_and_replacement_use_original_positions(kind, reverse, action):
    models = components(kind)
    stack = make_stack(kind, models)
    _, names = root_and_names(stack, kind)
    expected_models = [clone(model) for model in models]
    replacement = (
        LogisticRegression(max_iter=200, random_state=5)
        if "classifier" in kind
        else KNeighborsRegressor(n_neighbors=3)
    )
    changes = [(names[0], None)]
    if action == "remove_last":
        changes.append((names[2], None))
        expected_models = [expected_models[1]]
        parameter = "var_smoothing" if "classifier" in kind else "alpha"
        changes.append((names[1] + "__" + parameter, 0.25))
        expected_models[0].set_params(**{parameter: 0.25})
    elif action == "replace_middle":
        changes.append((names[1], replacement))
        expected_models = [clone(replacement), expected_models[2]]
    else:
        changes.append((names[2], replacement))
        expected_models = [expected_models[1], clone(replacement)]
    if reverse:
        changes.reverse()

    assert stack.set_params(**dict(changes)) is stack
    assert_fitted_equivalent(stack, make_stack(kind, expected_models), kind)


@pytest.mark.parametrize("kind", KINDS)
def test_root_replacement_precedes_multiple_named_removals(kind):
    models = components(kind)
    stack = make_stack(kind, [clone(models[0])])
    reference = make_stack(kind, models)
    root, names = root_and_names(reference, kind)
    stack.set_params(**{root: models, names[0]: None, names[2]: None})
    assert_fitted_equivalent(stack, make_stack(kind, [clone(models[1])]), kind)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_numbered_names_remain_stable_during_multiple_removals(kind, reverse):
    if "classifier" in kind:
        models = [
            DecisionTreeClassifier(max_depth=depth, random_state=17)
            for depth in (1, 2, 3)
        ]
    else:
        models = [Ridge(alpha=alpha) for alpha in (0.1, 1.0, 10.0)]
    stack = make_stack(kind, models)
    _, names = root_and_names(stack, kind)
    changes = [(names[0], None), (names[2], None)]
    if reverse:
        changes.reverse()
    stack.set_params(**dict(changes))
    assert_fitted_equivalent(stack, make_stack(kind, [clone(models[1])]), kind)


@pytest.mark.parametrize("kind", KINDS)
def test_grid_search_removals_match_explicit_remaining_models(kind):
    models = components(kind)
    stack = make_stack(kind, models)
    _, names = root_and_names(stack, kind)
    parameter = "var_smoothing" if "classifier" in kind else "alpha"
    nested = names[1] + "__" + parameter
    grid = {names[0]: [None], names[2]: [None], nested: [0.1, 1.0]}
    samples, targets = data(kind)
    scoring = "accuracy" if "classifier" in kind else "neg_mean_squared_error"
    kwargs = {"cv": 3, "scoring": scoring, "error_score": "raise"}
    actual = GridSearchCV(stack, grid, **kwargs).fit(samples, targets)
    reference = GridSearchCV(
        make_stack(kind, [clone(models[1])]), {nested: [0.1, 1.0]}, **kwargs
    ).fit(samples, targets)

    assert actual.best_params_[nested] == reference.best_params_[nested]
    for key, expected in reference.cv_results_.items():
        if key.endswith("_test_score"):
            np.testing.assert_allclose(actual.cv_results_[key], expected)
    np.testing.assert_allclose(actual.predict(samples), reference.predict(samples))
