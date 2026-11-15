from itertools import permutations

import numpy as np
import pytest
from scipy.linalg import eigh

from mlxtend.feature_extraction import LinearDiscriminantAnalysis

SAMPLES = np.array(
    [
        [0.0, 0.0],
        [2.0, 0.0],
        [4.0, 1.0],
        [4.0, 3.0],
        [6.0, 2.0],
        [0.0, 6.0],
        [2.0, 6.0],
        [0.0, 8.0],
        [2.0, 8.0],
    ]
)
LABELS = np.repeat([0, 1, 2], [2, 3, 4])

# Hand-computed scatter matrices for the three class means
# (1, 0), (14/3, 2), (1, 7), with class sizes 2, 3, 4.
WITHIN = np.diag([26 / 3, 6])
BETWEEN = np.array([[242 / 9, -176 / 9], [-176 / 9, 716 / 9]])


@pytest.mark.parametrize("relabeling", list(permutations(range(3))))
@pytest.mark.parametrize("n_discriminants", [1, 2])
def test_lda_uses_matching_class_sizes(relabeling, n_discriminants):
    samples = SAMPLES.copy()
    labels = np.asarray(relabeling)[LABELS]
    original_labels = labels.copy()
    expected_values, expected_vectors = eigh(BETWEEN, WITHIN)
    expected_values = expected_values[::-1]
    expected_vectors = expected_vectors[:, ::-1]
    expected_vectors /= np.linalg.norm(expected_vectors, axis=0)

    model = LinearDiscriminantAnalysis(n_discriminants=n_discriminants).fit(
        samples, labels
    )

    np.testing.assert_allclose(model.e_vals_, expected_values, rtol=1e-12)
    directions = expected_vectors[:, :n_discriminants]
    alignment = np.sign(np.sum(model.w_ * directions, axis=0))
    np.testing.assert_allclose(
        model.w_ * alignment, directions, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        model.transform(samples) * alignment,
        samples @ directions,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(samples, SAMPLES)
    np.testing.assert_array_equal(labels, original_labels)


def test_one_class_keeps_zero_between_class_scatter():
    samples = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    model = LinearDiscriminantAnalysis(n_discriminants=2).fit(
        samples, np.zeros(len(samples), dtype=int)
    )

    np.testing.assert_array_equal(model.e_vals_, [0, 0])
    assert np.isfinite(model.transform(samples)).all()
