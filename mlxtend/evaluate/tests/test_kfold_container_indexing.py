import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from mlxtend.evaluate import paired_ttest_kfold_cv


def regression_data():
    rng = np.random.RandomState(19)
    samples = rng.normal(size=(30, 4))
    targets = samples @ np.array([2.0, -0.5, 1.0, 0.25])
    targets += 0.3 * samples[:, 0] ** 2
    return samples, targets


def expected_result(samples, targets, shuffle):
    scores = [[], []]
    splitter = KFold(n_splits=5, shuffle=shuffle, random_state=17 if shuffle else None)
    scorer = get_scorer("neg_mean_squared_error")
    for train, test in splitter.split(samples):
        for column, alpha in enumerate([0.1, 20.0]):
            model = Ridge(alpha=alpha, solver="lsqr").fit(
                samples[train], targets[train]
            )
            scores[column].append(scorer(model, samples[test], targets[test]))
    result = stats.ttest_rel(*scores)
    assert np.isfinite([result.statistic, result.pvalue]).all()
    return result.statistic, result.pvalue


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(
    "container",
    [
        "array",
        "list",
        "dataframe",
        "series_reordered",
        "series_shifted",
        "series_duplicate",
        "csr",
        "csc",
        "coo",
    ],
)
def test_kfold_indexes_samples_by_position(container, shuffle):
    samples, targets = regression_data()
    expected = expected_result(samples, targets, shuffle)
    x_input, y_input = samples, targets
    if container == "list":
        x_input, y_input = samples.tolist(), targets.tolist()
    elif container == "dataframe":
        x_input = pd.DataFrame(samples, columns=["a", "b", "c", "d"])
        y_input = pd.Series(targets, index=np.arange(len(targets)) + 100)
    elif container == "series_reordered":
        y_input = pd.Series(targets, index=np.arange(len(targets))[::-1])
    elif container == "series_shifted":
        y_input = pd.Series(targets, index=np.arange(len(targets)) + 100)
    elif container == "series_duplicate":
        y_input = pd.Series(targets, index=np.zeros(len(targets), dtype=int))
    elif container in {"csr", "csc", "coo"}:
        x_input = getattr(sparse, container + "_matrix")(samples)

    actual = paired_ttest_kfold_cv(
        Ridge(alpha=0.1, solver="lsqr"),
        Ridge(alpha=20.0, solver="lsqr"),
        x_input,
        y_input,
        cv=5,
        scoring="neg_mean_squared_error",
        shuffle=shuffle,
        random_seed=17,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    if isinstance(x_input, pd.DataFrame):
        np.testing.assert_array_equal(x_input.to_numpy(), samples)
    if isinstance(y_input, pd.Series):
        np.testing.assert_array_equal(y_input.to_numpy(), targets)


def test_kfold_preserves_dataframe_columns_for_pipelines():
    samples, targets = regression_data()
    frame = pd.DataFrame(samples, columns=["a", "b", "c", "d"])
    target = pd.Series(targets, index=np.arange(len(targets))[::-1])
    models = [
        make_pipeline(
            ColumnTransformer([("columns", "passthrough", ["a", "c"])]),
            Ridge(alpha=alpha, solver="lsqr"),
        )
        for alpha in [0.1, 20.0]
    ]
    scores = [[], []]
    scorer = get_scorer("neg_mean_squared_error")
    for train, test in KFold(n_splits=5).split(frame):
        for column, model in enumerate(models):
            model.fit(frame.iloc[train], target.iloc[train])
            scores[column].append(scorer(model, frame.iloc[test], target.iloc[test]))
    expected = stats.ttest_rel(*scores)

    actual = paired_ttest_kfold_cv(*models, frame, target, cv=5, scoring=scorer)

    np.testing.assert_allclose(
        actual, [expected.statistic, expected.pvalue], rtol=1e-10, atol=1e-12
    )
