# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# License: BSD 3 clause

import numpy as np
import pytest

from mlxtend.classifier import (
    LogisticRegression,
    MultiLayerPerceptron,
    SoftmaxRegression,
)


@pytest.mark.parametrize("magnitude", [40.0, 1000.0])
@pytest.mark.parametrize("correct", [True, False])
@pytest.mark.parametrize("regularization", [0.0, 0.25])
def test_logistic_loss_stays_finite(magnitude, correct, regularization):
    features = np.array([[-1.0], [1.0], [0.0]])
    target = np.array([0, 1, 0]) if correct else np.array([1, 0, 1])
    model = LogisticRegression(epochs=0, eta=0, l2_lambda=regularization)
    model.fit(features, target)
    model.w_[:] = magnitude
    model.b_[:] = 0
    model.epochs = 1

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        model.fit(features, target, init_params=False)
        probabilities = model.predict_proba(features)

    expected_loss = 2 * (np.log1p(np.exp(-magnitude))) + np.log(2)
    if not correct:
        expected_loss += 2 * magnitude
    expected_loss += regularization * magnitude**2 / 2
    np.testing.assert_allclose(model.cost_[-1], expected_loss, rtol=1e-12)
    assert np.isfinite(probabilities).all()


@pytest.mark.parametrize("classifier", [SoftmaxRegression, MultiLayerPerceptron])
@pytest.mark.parametrize("magnitude", [40.0, 1000.0])
@pytest.mark.parametrize("offset", [0.0, 10000.0])
@pytest.mark.parametrize("target_class", [0, 2])
def test_multiclass_loss_uses_logits(classifier, magnitude, offset, target_class):
    features = np.array([[0.0], [1.0], [-1.0]])
    target = np.full(3, target_class, dtype=int)
    model = classifier(epochs=0, eta=0, n_classes=3)
    model.fit(features, target)
    logits = np.array([magnitude, 0.0, -magnitude]) + offset
    if isinstance(model, SoftmaxRegression):
        model.w_[:] = 0
        model.b_[:] = logits
    else:
        model.w_["out"][:] = 0
        model.b_["out"][:] = logits
    model.epochs = 1

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        model.fit(features, target, init_params=False)

    # The existing objective includes a factor of one half. A confidently
    # wrong label has loss proportional to the logit gap, not a clipped value.
    expected_loss = np.log1p(np.exp(-magnitude) + np.exp(-2 * magnitude))
    if target_class == 2:
        expected_loss += 2 * magnitude
    np.testing.assert_allclose(model.cost_[-1], expected_loss / 2, atol=1e-14)
