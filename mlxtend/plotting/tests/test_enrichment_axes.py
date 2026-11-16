# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mlxtend.plotting import enrichment_plot  # noqa: E402


@pytest.fixture
def axes():
    fig, axes = plt.subplots(1, 2)
    yield axes
    plt.close(fig)


@pytest.mark.parametrize("invert", [False, True])
def test_supplied_axes_owns_all_plot_elements(axes, invert):
    target, current = axes
    current.plot([1, 2], [3, 4], label="other subplot")
    current.set_xlabel("other x")
    current.set_ylabel("other y")
    current.legend()
    other_legend = current.get_legend()
    plt.sca(current)
    frame = pd.DataFrame({"A": [3.0, -2.0, 1.0], "B": [8.0, 7.0, 5.0]})
    original = frame.copy(deep=True)

    result = enrichment_plot(frame, invert_axes=invert, ax=target, count_label="N")

    assert result is target
    assert plt.gca() is current
    assert [text.get_text() for text in target.get_legend().texts] == ["A", "B"]
    assert target.get_xlabel() == ("N" if invert else "")
    assert target.get_ylabel() == ("" if invert else "N")
    assert all(line.get_visible() for line in target.get_xgridlines())
    assert all(line.get_visible() for line in target.get_ygridlines())
    assert current.get_legend() is other_legend
    assert current.get_xlabel() == "other x"
    assert current.get_ylabel() == "other y"
    assert not any(line.get_visible() for line in current.get_xgridlines())
    assert not any(line.get_visible() for line in current.get_ygridlines())
    assert len(current.lines) == 1
    for line, values in zip(target.lines, [(-2, 1, 3), (5, 7, 8)]):
        expected_x, expected_y = (1, 2, 3), values
        if not invert:
            expected_x, expected_y = expected_y, expected_x
        np.testing.assert_array_equal(line.get_xdata(), expected_x)
        np.testing.assert_array_equal(line.get_ydata(), expected_y)
    pd.testing.assert_frame_equal(frame, original)
    target.figure.canvas.draw()


@pytest.mark.parametrize("invert", [False, True])
def test_axis_methods_retain_their_standard_meaning(axes, invert):
    target = axes[0]
    original_xlim = target.set_xlim
    original_ylim = target.set_ylim
    enrichment_plot(pd.Series([2, 4, 3], name="data"), ax=target, invert_axes=invert)

    assert target.set_xlim == original_xlim
    assert target.set_ylim == original_ylim
    previous_y = target.get_ylim()
    target.set_xlim(101, 109)
    assert target.get_xlim() == (101, 109)
    assert target.get_ylim() == previous_y
    target.set_ylim(-8, -2)
    assert target.get_ylim() == (-8, -2)
    assert target.get_xlim() == (101, 109)


@pytest.mark.parametrize("sequence", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("explicit", [False, True])
def test_reusing_axes_keeps_value_and_count_limits(axes, sequence, explicit):
    target = axes[0]
    frame = pd.DataFrame({"A": [3, 6, 9]})
    kwargs = {"xlim": (2, 10), "ylim": (-2, 5)} if explicit else {}
    value_limits = (2, 10)
    count_limits = (-2, 5) if explicit else (0, 4)
    for invert in sequence:
        enrichment_plot(
            frame, ax=target, invert_axes=invert, grid=False, legend_loc=False, **kwargs
        )
        assert target.get_xlim() == (count_limits if invert else value_limits)
        assert target.get_ylim() == (value_limits if invert else count_limits)
        target.figure.canvas.draw()


@pytest.mark.parametrize("invert", [False, True])
def test_default_axes_and_disabled_optional_elements(axes, invert):
    target = axes[0]
    plt.sca(target)
    target.set_xlabel("existing x")
    target.set_ylabel("existing y")
    result = enrichment_plot(
        pd.Series([1, 2, 3]),
        invert_axes=invert,
        grid=False,
        count_label="",
        legend_loc=False,
    )
    assert result is target
    assert target.get_legend() is None
    assert target.get_xlabel() == "existing x"
    assert target.get_ylabel() == "existing y"
    assert not any(line.get_visible() for line in target.get_xgridlines())
    assert not any(line.get_visible() for line in target.get_ygridlines())
