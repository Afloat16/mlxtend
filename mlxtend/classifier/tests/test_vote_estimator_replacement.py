import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import EnsembleVoteClassifier


def data():
    samples = np.arange(36).reshape(-1, 1)
    targets = np.array([0] * 12 + [1] * 24)
    return samples, targets


@pytest.mark.parametrize("voting", ["hard", "soft"])
@pytest.mark.parametrize("use_clones", [False, True])
def test_named_replacement_is_used_for_fitting(voting, use_clones):
    samples, targets = data()
    original = DummyClassifier(strategy="constant", constant=0)
    replacement = DummyClassifier(strategy="constant", constant=1)
    model = EnsembleVoteClassifier([original], voting=voting, use_clones=use_clones)

    result = model.set_params(dummyclassifier=replacement).fit(samples, targets)

    assert result is model
    np.testing.assert_array_equal(model.predict(samples), np.ones(len(samples)))
    assert model.clfs[0] is replacement
    assert model.named_clfs["dummyclassifier"] is replacement
    assert original.constant == 0
    assert not hasattr(original, "classes_")


@pytest.mark.parametrize("voting", ["hard", "soft"])
@pytest.mark.parametrize("together", [False, True])
def test_replaced_list_receives_nested_parameters(voting, together):
    samples, targets = data()
    original = DummyClassifier(strategy="constant", constant=0)
    replacement = DummyClassifier(strategy="constant", constant=0)
    model = EnsembleVoteClassifier([original], voting=voting)

    if together:
        model.set_params(clfs=[replacement], dummyclassifier__constant=1)
    else:
        model.set_params(clfs=[replacement])
        model.set_params(dummyclassifier__constant=1)
    model.fit(samples, targets)

    np.testing.assert_array_equal(model.predict(samples), np.ones(len(samples)))
    assert replacement.constant == 1
    assert original.constant == 0
    assert model.get_params()["dummyclassifier"] is replacement
    copied = clone(model).fit(samples, targets)
    np.testing.assert_array_equal(copied.predict(samples), model.predict(samples))


@pytest.mark.parametrize("voting", ["hard", "soft"])
@pytest.mark.parametrize("together", [False, True])
def test_new_classifier_names_after_list_replacement(voting, together):
    samples, targets = data()
    original = GaussianNB()
    replacement = DecisionTreeClassifier(random_state=3)
    model = EnsembleVoteClassifier([original], voting=voting)

    if together:
        model.set_params(clfs=[replacement], decisiontreeclassifier__max_depth=1)
    else:
        model.set_params(clfs=[replacement])
        model.set_params(decisiontreeclassifier__max_depth=1)
    model.fit(samples, targets)

    np.testing.assert_array_equal(model.predict(samples), targets)
    assert replacement.max_depth == 1
    assert "gaussiannb" not in model.get_params()
    assert model.named_clfs == {"decisiontreeclassifier": replacement}
    assert not hasattr(original, "classes_")


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_named_replacement_then_parameters_of_new_type(voting):
    samples, targets = data()
    replacement = DecisionTreeClassifier(random_state=3)
    model = EnsembleVoteClassifier([GaussianNB()], voting=voting)
    model.set_params(gaussiannb=replacement, decisiontreeclassifier__max_depth=1)
    model.fit(samples, targets)

    np.testing.assert_array_equal(model.predict(samples), targets)
    assert model.clfs[0] is replacement
    assert model.clfs_[0].max_depth == 1


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_numbered_classifier_replacement_preserves_vote_weights(voting):
    samples, targets = data()
    original = [
        DummyClassifier(strategy="constant", constant=0),
        DummyClassifier(strategy="constant", constant=0),
    ]
    replacement = DummyClassifier(strategy="constant", constant=1)
    model = EnsembleVoteClassifier(original, voting=voting, weights=[1, 2])
    model.set_params(**{"dummyclassifier-2": replacement}).fit(samples, targets)

    np.testing.assert_array_equal(model.predict(samples), np.ones(len(samples)))
    np.testing.assert_allclose(
        model.predict_proba(samples), np.tile([1 / 3, 2 / 3], (len(samples), 1))
    )
    assert original[1].constant == 0
    assert model.clfs[0] is original[0]
    assert model.clfs[1] is replacement


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_grid_search_scores_the_replacement_classifiers(voting):
    samples, targets = data()
    model = EnsembleVoteClassifier(
        [DummyClassifier(strategy="constant", constant=0)], voting=voting
    )
    candidates = [
        DummyClassifier(strategy="constant", constant=0),
        DummyClassifier(strategy="constant", constant=1),
    ]
    search = GridSearchCV(
        model,
        {"dummyclassifier": candidates},
        scoring="accuracy",
        cv=StratifiedKFold(3, shuffle=True, random_state=7),
        error_score="raise",
    ).fit(samples, targets)

    np.testing.assert_allclose(search.cv_results_["mean_test_score"], [1 / 3, 2 / 3])
    np.testing.assert_array_equal(search.predict(samples), np.ones(len(samples)))
    assert search.best_estimator_.clfs_[0].constant == 1
    assert model.clfs[0].constant == 0


@pytest.mark.parametrize("voting", ["hard", "soft"])
def test_nested_parameter_control_without_replacement(voting):
    samples, targets = data()
    original = DummyClassifier(strategy="constant", constant=0)
    model = EnsembleVoteClassifier([original], voting=voting)
    model.set_params(dummyclassifier__constant=1).fit(samples, targets)

    np.testing.assert_array_equal(model.predict(samples), np.ones(len(samples)))
    assert original.constant == 1
