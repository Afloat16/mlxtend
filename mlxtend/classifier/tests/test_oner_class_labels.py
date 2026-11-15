import numpy as np
import pytest
from scipy.stats import chi2_contingency

from mlxtend.classifier import OneRClassifier

LABEL_SETS = [
    np.array([0, 1, 2]),
    np.array([0, 3, 5]),
    np.array([-20, -4, 10]),
    np.array([2**63 + 1, 2**63 + 5, 2**63 + 9], dtype=np.uint64),
    np.array(["ant", "bee", "horse"]),
    np.array([1.0, 5.0, 9.0]),
]


@pytest.mark.parametrize("labels", LABEL_SETS)
@pytest.mark.parametrize("resolve_ties", ["first", "chi-squared"])
@pytest.mark.parametrize("unused_class", [False, True])
def test_oner_retains_original_class_labels(labels, resolve_ties, unused_class):
    if unused_class:
        samples = np.array([[0], [0], [0], [1], [1], [1]])
        encoded = np.array([0, 0, 1, 2, 2, 1])
        query = np.array([[0], [1], [9]])
        expected_codes = [0, 2, 1]
        expected_error = 2
    else:
        samples = np.array([[0], [0], [0], [1], [1], [2], [2], [2]])
        encoded = np.array([0, 0, 1, 1, 1, 2, 2, 2])
        query = np.array([[0], [2], [1], [9]])
        expected_codes = [0, 2, 1, 0]
        expected_error = 1
    targets = labels[encoded]
    original_targets = targets.copy()

    model = OneRClassifier(resolve_ties=resolve_ties).fit(samples, targets)

    np.testing.assert_array_equal(model.predict(query), labels[expected_codes])
    np.testing.assert_array_equal(model.class_labels_, labels)
    assert model.predict(query).dtype == labels.dtype
    assert model.feature_idx_ == 0
    assert model.prediction_dict_["total error"] == expected_error
    rules = model.prediction_dict_["rules (value: class)"]
    assert set(rules.values()).issubset(set(labels))
    if resolve_ties == "chi-squared":
        contingency = np.array(
            [
                [
                    np.count_nonzero((encoded == label) & (samples[:, 0] == value))
                    for value in np.unique(samples)
                ]
                for label in range(len(labels))
            ]
        )
        expected_p = chi2_contingency(contingency)[1]
        np.testing.assert_allclose(model.p_value_, expected_p)
    np.testing.assert_array_equal(targets, original_targets)


def test_oner_refit_replaces_the_label_mapping():
    samples = np.array([[0], [0], [1], [1]])
    model = OneRClassifier().fit(samples, np.array([-2, -2, 7, 7]))

    model.fit(samples, np.array(["long_name", "long_name", "z", "z"]))

    np.testing.assert_array_equal(
        model.predict(np.array([[0], [1], [9]])),
        ["long_name", "z", "long_name"],
    )
    assert model.prediction_dict_["rules (value: class)"] == {0: "long_name", 1: "z"}
