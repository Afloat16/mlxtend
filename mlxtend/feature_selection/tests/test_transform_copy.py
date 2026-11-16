# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def _selector(kind):
    kwargs = dict(
        estimator=LinearRegression(),
        scoring="neg_mean_squared_error",
        cv=0,
        fixed_features=(0, 2),
    )
    if kind == "sfs":
        return SFS(k_features=2, **kwargs)
    return EFS(min_features=2, max_features=2, print_progress=False, **kwargs)


def _input(kind, values):
    if kind == "dataframe":
        return pd.DataFrame(values, index=np.arange(len(values)) + 100)
    if kind == "mixed_dataframe":
        return pd.DataFrame(
            {"a": values[:, 0], "b": values[:, 1].astype(int), "c": values[:, 2]}
        )
    if kind == "csr":
        return sparse.csr_matrix(values)
    if kind == "csc":
        return sparse.csc_matrix(values)
    if kind == "strided":
        return np.repeat(values, 2, axis=1)[:, ::2]
    result = values.copy(order="F" if kind == "fortran" else "C")
    if kind == "readonly":
        result.flags.writeable = False
    return result


@pytest.mark.parametrize("selector_kind", ["sfs", "efs"])
@pytest.mark.parametrize(
    "input_kind",
    [
        "array",
        "fortran",
        "strided",
        "readonly",
        "dataframe",
        "mixed_dataframe",
        "csr",
        "csc",
    ],
)
def test_transform_matches_selected_columns_and_owns_output(selector_kind, input_kind):
    rng = np.random.RandomState(71)
    values = rng.normal(size=(18, 3))
    X = _input(input_kind, values)
    y = 2 * values[:, 0] - 3 * values[:, 2]
    selector = _selector(selector_kind).fit(X, y)
    result = selector.transform(X)
    if sparse.issparse(X):
        expected = X.toarray()[:, [0, 2]]
        assert sparse.issparse(result)
        assert result.format == X.format
        np.testing.assert_array_equal(result.toarray(), expected)
        result.data[:] = 123
        np.testing.assert_array_equal(X.toarray(), values)
    else:
        expected = np.asarray(X)[:, [0, 2]]
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.asarray(X).dtype
        np.testing.assert_array_equal(result, expected)
        result[:] = 123
        np.testing.assert_array_equal(np.asarray(X)[:, [0, 2]], expected)


@pytest.mark.parametrize("selector_kind", ["sfs", "efs"])
@pytest.mark.parametrize("input_kind", ["array", "dataframe", "csr", "csc"])
def test_real_pipeline_predictions_match_manually_selected_data(
    selector_kind, input_kind
):
    rng = np.random.RandomState(3)
    values = rng.normal(size=(30, 3))
    target = values[:, 0] - 2 * values[:, 2] + rng.normal(scale=0.1, size=30)
    X = _input(input_kind, values)
    pipeline = make_pipeline(_selector(selector_kind), LinearRegression())
    pipeline.fit(X, target)
    manual = clone(pipeline[-1]).fit(values[:, [0, 2]], target)
    test_values = rng.normal(size=(7, 3))
    actual = pipeline.predict(_input(input_kind, test_values))
    np.testing.assert_allclose(actual, manual.predict(test_values[:, [0, 2]]))


@pytest.mark.parametrize("selector_kind", ["sfs", "efs"])
def test_dataframe_output_keeps_common_dtype_and_positional_labels(selector_kind):
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
    selector = _selector(selector_kind).fit(X, np.arange(3))
    mixed = pd.DataFrame([[1.0, "unused", 3.0], [4.0, "other", 6.0]])
    mixed.columns = ["duplicate", "duplicate", "different"]
    result = selector.transform(mixed)
    assert result.dtype == object
    np.testing.assert_array_equal(
        result, np.array([[1.0, 3.0], [4.0, 6.0]], dtype=object)
    )
    result[0, 0] = "changed"
    assert mixed.iloc[0, 0] == 1.0


@pytest.mark.parametrize("selector_kind", ["sfs", "efs"])
def test_transform_preserves_existing_dimension_and_fitted_checks(selector_kind):
    selector = _selector(selector_kind)
    with pytest.raises(AttributeError, match="fitted"):
        selector.transform(np.ones((3, 3)))
    selector.fit(np.arange(15).reshape(5, 3), np.arange(5))
    with pytest.raises(ValueError, match="2D"):
        selector.transform(np.arange(3))
    assert selector.transform(np.empty((0, 3))).shape == (0, 2)
