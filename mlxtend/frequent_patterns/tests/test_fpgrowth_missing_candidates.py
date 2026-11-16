"""Missing-value support cannot use ordinary downward-closure pruning."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mlxtend.frequent_patterns import association_rules, fpgrowth


def exhaustive_supports(frame, threshold, use_colnames=False, max_len=None):
    values = frame.to_numpy()
    expected = {}
    for size in range(1, (max_len or frame.shape[1]) + 1):
        for indices in combinations(range(frame.shape[1]), size):
            subset = values[:, indices]
            observed = subset[~pd.isna(subset).any(axis=1)]
            support = (observed == 1).all(axis=1).mean() if len(observed) else 0
            if support >= threshold:
                items = frame.columns[list(indices)] if use_colnames else indices
                expected[frozenset(items)] = support
    return expected


def actual_supports(frame, threshold, use_colnames=False, max_len=None):
    result = fpgrowth(
        frame,
        min_support=threshold,
        null_values=True,
        use_colnames=use_colnames,
        max_len=max_len,
    )
    assert len(result) == len(set(result["itemsets"]))
    return {row.itemsets: row.support for row in result.itertuples()}


@pytest.mark.parametrize("use_colnames", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        [[1, 1], [0, np.nan], [0, np.nan], [0, np.nan]],
        [[1, 1, 0], [1, 0, 1], [0, np.nan, 1], [0, np.nan, 1]],
        [[1, 1, np.nan], [0, np.nan, 1], [np.nan, 1, 1]],
        [[1, 1, np.nan], [0, np.nan, np.nan], [0, np.nan, np.nan]],
    ],
)
def test_missing_value_itemsets_match_complete_row_enumeration(values, use_colnames):
    frame = pd.DataFrame(values, columns=list("ABC")[: len(values[0])])
    assert actual_supports(frame, 0.5, use_colnames) == exhaustive_supports(
        frame, 0.5, use_colnames
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("threshold", [0.5, 0.75])
def test_small_random_tables_match_all_itemsets(seed, threshold):
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame(rng.choice([0.0, 1.0, np.nan], size=(6, 4)))
    assert actual_supports(frame, threshold) == exhaustive_supports(frame, threshold)


@pytest.mark.parametrize("threshold", [0.5, 1.0])
def test_fully_observed_data_keeps_the_existing_result(threshold):
    frame = pd.DataFrame([[True, True], [False, True], [True, False]])
    result = fpgrowth(frame, min_support=threshold)
    expected = {row.itemsets: row.support for row in result.itertuples()}
    assert actual_supports(frame, threshold) == expected


def test_all_missing_columns_have_no_frequent_itemsets():
    frame = pd.DataFrame(np.full((3, 2), np.nan))
    assert actual_supports(frame, 0.5) == {}


def test_maximum_length_remains_a_search_bound():
    frame = pd.DataFrame(
        [[1, 1, 1], [0, np.nan, np.nan], [np.nan, 1, 1]], columns=list("ABC")
    )
    assert actual_supports(frame, 0.5, max_len=2) == exhaustive_supports(
        frame, 0.5, max_len=2
    )


@pytest.mark.parametrize("use_colnames", [False, True])
def test_rules_recover_infrequent_subset_supports(use_colnames):
    frame = pd.DataFrame(
        [[1, 1], [0, np.nan], [0, np.nan], [0, np.nan]], columns=["A", "B"]
    )
    original = frame.copy(deep=True)
    itemsets = fpgrowth(
        frame, min_support=0.5, null_values=True, use_colnames=use_colnames
    )
    rules = association_rules(
        itemsets,
        num_itemsets=len(frame),
        df_orig=frame,
        null_values=True,
        min_threshold=0.5,
        return_metrics=[
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "representativity",
        ],
    )
    a, b = ("A", "B") if use_colnames else (0, 1)
    # A occurs in one of four observed rows; B and AB occur in the only
    # row where B is observed. Both directions therefore have confidence 1.
    expected = {
        (frozenset([a]), frozenset([b])): [0.25, 1, 1, 1, 1, 0.25],
        (frozenset([b]), frozenset([a])): [1, 0.25, 1, 1, 4, 0.25],
    }
    assert len(rules) == 2
    for row in rules.itertuples(index=False, name=None):
        assert list(row[2:]) == expected[row[:2]]
    pd.testing.assert_frame_equal(frame, original)


def test_rules_recover_multi_item_subsets_without_adding_infrequent_rules():
    frame = pd.DataFrame(
        [[1, 1, 1], [0, 1, np.nan], [1, 0, np.nan], [0, 0, np.nan]],
        columns=["A", "B", "C"],
    )
    itemsets = fpgrowth(frame, min_support=0.75, null_values=True, use_colnames=True)
    rules = association_rules(
        itemsets,
        num_itemsets=len(frame),
        df_orig=frame,
        null_values=True,
        min_threshold=0.5,
        return_metrics=["antecedent support", "consequent support", "confidence"],
    )
    # AC and BC each give two rules, and ABC gives six. AB is infrequent
    # (one positive row out of four), so recovering its support adds no rules.
    assert len(rules) == 10
    supports = {
        frozenset("A"): 0.5,
        frozenset("B"): 0.5,
        frozenset("C"): 1,
        frozenset("AB"): 0.25,
        frozenset("AC"): 1,
        frozenset("BC"): 1,
    }
    for antecedent, consequent, s_a, s_c, confidence in rules.itertuples(
        index=False, name=None
    ):
        assert s_a == supports[antecedent]
        assert s_c == supports[consequent]
        assert confidence == 1
        assert "C" in antecedent | consequent


def test_missing_supports_still_raise_without_missing_value_mode():
    incomplete = pd.DataFrame({"support": [1.0], "itemsets": [frozenset("AB")]})
    with pytest.raises(KeyError, match="missing.*antecedent"):
        association_rules(incomplete, num_itemsets=4)


def test_missing_value_rules_still_require_original_data():
    incomplete = pd.DataFrame({"support": [1.0], "itemsets": [frozenset("AB")]})
    with pytest.raises(TypeError, match="df_orig must be provided"):
        association_rules(incomplete, num_itemsets=4, null_values=True)
