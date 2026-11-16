# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from mlxtend.feature_extraction import PrincipalComponentAnalysis  # noqa: E402
from mlxtend.plotting import plot_pca_correlation_graph  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _data(dtype=np.float64):
    rng = np.random.RandomState(17)
    X = rng.normal(size=(40, 6)) @ rng.normal(size=(6, 6))
    return (X + np.arange(6)).astype(dtype)


def _correlations(X, scores, count):
    return np.array(
        [
            [
                pearsonr(X[:, feature], scores[:, component]).statistic
                for component in range(count)
            ]
            for feature in range(X.shape[1])
        ]
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "components, dimensions",
    [
        (2, (1, 2)),
        (3, (1, 2)),
        (6, (1, 2)),
        (6, (4, 2)),
        (6, (3, 5)),
        (3, (2, 3)),
        (6, (2, 1)),
    ],
)
def test_full_precomputed_projection_matches_pearson_correlations(
    dtype, components, dimensions
):
    X = _data(dtype)
    model = PCA(n_components=components, svd_solver="full").fit(X)
    scores = model.transform(X)
    eigenvalues = model.explained_variance_.copy()
    before = X.copy(), scores.copy(), eigenvalues.copy()
    scores.flags.writeable = False
    names = [f"feature_{i}" for i in range(X.shape[1])]
    fig, table = plot_pca_correlation_graph(
        X,
        names,
        dimensions=dimensions,
        X_pca=scores,
        explained_variance=eigenvalues,
        figure_axis_size=2,
    )
    count = max(dimensions)
    expected = _correlations(X, scores, count)
    assert table.shape == (X.shape[1], count)
    assert list(table.index) == names
    assert list(table.columns) == [f"Dim {i + 1}" for i in range(count)]
    np.testing.assert_allclose(table.to_numpy(), expected, rtol=2e-5, atol=2e-6)
    ax = fig.axes[0]
    for feature, line in enumerate(ax.lines[: X.shape[1]]):
        np.testing.assert_allclose(
            line.get_xdata(), [0, expected[feature, dimensions[0] - 1]], atol=2e-6
        )
        np.testing.assert_allclose(
            line.get_ydata(), [0, expected[feature, dimensions[1] - 1]], atol=2e-6
        )
    assert ax.get_xlabel().startswith(f"Dim {dimensions[0]} (")
    assert ax.get_ylabel().startswith(f"Dim {dimensions[1]} (")
    for actual, original in zip((X, scores, eigenvalues), before):
        np.testing.assert_array_equal(actual, original)
    fig.canvas.draw()


def test_precomputed_component_signs_and_variance_labels_are_preserved():
    X = _data()
    model = PCA(n_components=6).fit(X)
    scores = model.transform(X) * np.array([-1, 1, -1, 1, -1, 1])
    fig, actual = plot_pca_correlation_graph(
        X,
        list("abcdef"),
        X_pca=scores,
        explained_variance=model.explained_variance_,
        figure_axis_size=2,
    )
    reference_fig, reference = plot_pca_correlation_graph(
        X,
        list("abcdef"),
        X_pca=scores[:, :2],
        explained_variance=model.explained_variance_[:2],
        figure_axis_size=2,
    )
    np.testing.assert_allclose(actual, _correlations(X, scores, 2), atol=1e-14)
    np.testing.assert_array_equal(actual, reference)
    assert fig.axes[0].get_xlabel() == reference_fig.axes[0].get_xlabel()
    assert fig.axes[0].get_ylabel() == reference_fig.axes[0].get_ylabel()


def test_automatic_pca_keeps_its_existing_component_table():
    X = _data()
    centered = X - X.mean(axis=0)
    scores = (
        PrincipalComponentAnalysis(n_components=2).fit(centered).transform(centered)
    )
    _, actual = plot_pca_correlation_graph(X, list("abcdef"), figure_axis_size=2)
    np.testing.assert_allclose(actual, _correlations(X, scores, 2), atol=1e-14)


@pytest.mark.parametrize(
    "problem", ["missing_variance", "mismatched_count", "too_few_components"]
)
def test_precomputed_input_validation_is_preserved(problem):
    X = _data()
    model = PCA(n_components=2).fit(X)
    kwargs = dict(
        X_pca=model.transform(X), explained_variance=model.explained_variance_
    )
    if problem == "missing_variance":
        kwargs["explained_variance"] = None
    elif problem == "mismatched_count":
        kwargs["explained_variance"] = model.explained_variance_[:1]
    else:
        kwargs["dimensions"] = (1, 3)
    with pytest.raises(ValueError):
        plot_pca_correlation_graph(X, list("abcdef"), **kwargs)
