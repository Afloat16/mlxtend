# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from mlxtend.plotting import ecdf  # noqa: E402


@pytest.mark.parametrize(
    "values, percentile, threshold, count",
    [
        ([1, 1, 1, 2], 0.5, 1, 3),
        ([0, 0, 0, 0], 0.75, 0, 4),
        ([1, 2, 2, 2, 3, 4], 0.5, 2, 4),
        ([-3, -3, -1, 0, 2, 2], 5 / 6, 2, 6),
        ([9, 2, 2, 4, 4, 4, 10, 10], 0.5, 4, 5),
        ([4, 1, 3, 2], 0.5, 2, 2),
        ([1, 1, 2, 3], 0.75, 2, 3),
        ([1, 1, 2, 3], 1.0, 3, 4),
    ],
)
@pytest.mark.parametrize("kind", ["list", "integer", "readonly_float"])
def test_percentile_count_includes_every_sample_at_the_threshold(
    values, percentile, threshold, count, kind
):
    if kind == "list":
        X = values.copy()
    elif kind == "integer":
        X = np.array(values, dtype=np.int64)
    else:
        X = np.array(values, dtype=np.float32)
        X.flags.writeable = False
    original = np.asarray(X).copy()
    fig, ax = plt.subplots()
    try:
        returned_ax, actual_threshold, actual_count = ecdf(
            X, percentile=percentile, ax=ax
        )
        assert returned_ax is ax
        assert actual_threshold == threshold
        assert actual_count == count
        assert isinstance(actual_count, int)
        assert actual_count == sum(value <= actual_threshold for value in values)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), sorted(values))
        np.testing.assert_array_equal(
            ax.lines[0].get_ydata(), np.arange(1, len(values) + 1) / len(values)
        )
        np.testing.assert_array_equal(ax.lines[1].get_xdata(), [threshold, threshold])
        np.testing.assert_array_equal(np.asarray(X), original)
        fig.canvas.draw()
    finally:
        plt.close(fig)


def test_absent_percentile_keeps_no_threshold_or_count():
    fig, ax = plt.subplots()
    try:
        result = ecdf([2, 1, 1, 1], ax=ax)
        assert result == (ax, None, None)
        assert len(ax.lines) == 1
    finally:
        plt.close(fig)
