import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from mlxtend.evaluate import feature_importance_permutation


def make_input(kind, tmp_path):
    rng = np.random.RandomState(19)
    owner = rng.normal(size=(40, 6))
    if kind == "view":
        samples = owner[:, ::2]
    else:
        samples = owner[:, :3].copy()
    if kind == "readonly":
        samples.setflags(write=False)
    elif kind == "memmap":
        path = tmp_path / "features.npy"
        np.save(path, samples)
        samples = np.load(path, mmap_mode="r")
    return samples, owner


def reference_importance(samples, targets, model, columns, rounds, seed):
    rng = np.random.RandomState(seed)
    baseline = r2_score(targets, model.predict(samples))
    values = np.empty((len(columns), rounds))
    for round_idx in range(rounds):
        for feature_idx, column in enumerate(columns):
            permuted = samples.copy()
            order = rng.permutation(len(samples))
            permuted[:, column] = samples[order, column]
            values[feature_idx, round_idx] = baseline - r2_score(
                targets, model.predict(permuted)
            )
    return values.mean(axis=1), values


@pytest.mark.parametrize("kind", ["writable", "readonly", "view", "memmap"])
@pytest.mark.parametrize("feature_groups", [None, [2, 0]])
def test_input_storage_is_preserved(kind, feature_groups, tmp_path):
    samples, owner = make_input(kind, tmp_path)
    before = samples.copy()
    owner_before = owner.copy()
    writable = samples.flags.writeable
    targets = 2 * samples[:, 0] - samples[:, 2]
    model = LinearRegression().fit(samples, targets)
    columns = (
        list(range(samples.shape[1])) if feature_groups is None else feature_groups
    )
    expected = reference_importance(samples, targets, model, columns, 4, 7)

    actual = feature_importance_permutation(
        samples,
        targets,
        model.predict,
        metric="r2",
        num_rounds=4,
        feature_groups=feature_groups,
        seed=7,
    )

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(samples, before)
    np.testing.assert_array_equal(owner, owner_before)
    assert samples.flags.writeable == writable
    if kind == "memmap":
        np.testing.assert_array_equal(np.load(tmp_path / "features.npy"), before)


@pytest.mark.parametrize("kind", ["writable", "view"])
@pytest.mark.parametrize("feature_groups", [None, [2, 0]])
@pytest.mark.parametrize("failing_callback", ["predict", "metric"])
def test_callback_failure_does_not_leave_permuted_input(
    kind, feature_groups, failing_callback, tmp_path
):
    samples, owner = make_input(kind, tmp_path)
    before = samples.copy()
    owner_before = owner.copy()
    targets = 2 * samples[:, 0] - samples[:, 2]
    model = LinearRegression().fit(samples, targets)
    predictions_before = model.predict(samples)
    failure = RuntimeError("callback stopped")
    calls = {"predict": 0, "metric": 0}

    def predict(values):
        result = model.predict(values)
        calls["predict"] += 1
        if failing_callback == "predict" and calls["predict"] == 2:
            raise failure
        return result

    def metric(true, predicted):
        result = r2_score(true, predicted)
        calls["metric"] += 1
        if failing_callback == "metric" and calls["metric"] == 2:
            raise failure
        return result

    with pytest.raises(RuntimeError) as exc:
        feature_importance_permutation(
            samples,
            targets,
            predict,
            metric=metric,
            num_rounds=4,
            feature_groups=feature_groups,
            seed=7,
        )

    assert exc.value is failure
    np.testing.assert_array_equal(samples, before)
    np.testing.assert_array_equal(owner, owner_before)
    np.testing.assert_array_equal(model.predict(samples), predictions_before)
