# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from mlxtend.frequent_patterns import association_rules

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:DataFrames with non-bool types:DeprecationWarning"
    ),
    pytest.mark.filterwarnings("ignore:null_values=True is inefficient:UserWarning"),
]


def _supports(data):
    """Enumerate itemsets using the documented complete-row denominator."""
    records = []
    for size in range(1, len(data.columns) + 1):
        for items in combinations(data.columns, size):
            observed = [
                row
                for row in data.loc[:, list(items)].itertuples(index=False, name=None)
                if not any(pd.isna(value) for value in row)
            ]
            support = sum(all(value == 1 for value in row) for row in observed) / len(
                observed
            )
            records.append((support, frozenset(items)))
    return pd.DataFrame(records, columns=["support", "itemsets"])


def _rule_metrics(data, antecedent, consequent):
    """Count eligible transactions directly, independently of disabled arrays."""
    rows = data.to_dict("records")

    def present(row, items):
        return all(row[item] == 1 for item in items)

    def observed(row, items):
        return all(not pd.isna(row[item]) for item in items)

    union = antecedent | consequent
    both = sum(present(row, union) for row in rows)
    a_count = sum(present(row, antecedent) for row in rows)
    c_count = sum(present(row, consequent) for row in rows)
    a_observed = sum(observed(row, antecedent) for row in rows)
    c_observed = sum(observed(row, consequent) for row in rows)
    both_observed = sum(observed(row, union) for row in rows)
    eligible_a = sum(
        present(row, antecedent) and observed(row, consequent) for row in rows
    )
    eligible_c = sum(
        present(row, consequent) and observed(row, antecedent) for row in rows
    )
    s_a, s_c, s_ac = (
        a_count / a_observed,
        c_count / c_observed,
        both / both_observed,
    )
    conf = both / eligible_a
    reverse_conf = both / eligible_c
    leverage = s_ac - s_a * s_c
    zhang_denom = max(s_ac * (1 - s_a), s_a * (s_c - s_ac))
    jaccard_denom = s_a + s_c - s_ac
    return {
        "antecedent support": s_a,
        "consequent support": s_c,
        "support": s_ac,
        "confidence": conf,
        "lift": conf / s_c,
        "representativity": both_observed / len(rows),
        "leverage": leverage,
        "conviction": (1 - s_c) / (1 - conf) if conf < 1 else np.inf,
        "zhangs_metric": leverage / zhang_denom if zhang_denom else 0,
        "jaccard": s_ac / jaccard_denom if jaccard_denom else np.inf,
        "certainty": (conf - s_c) / (1 - s_c) if s_c < 1 else 0,
        "kulczynski": (conf + reverse_conf) / 2,
    }


def _check_rules(data, metric="support", threshold=0, return_metrics=None):
    itemsets = _supports(data)
    original_data = data.copy(deep=True)
    original_itemsets = itemsets.copy(deep=True)
    kwargs = {} if return_metrics is None else {"return_metrics": return_metrics}
    with np.errstate(invalid="ignore", divide="ignore"):
        result = association_rules(
            itemsets,
            num_itemsets=len(data),
            df_orig=data,
            null_values=True,
            metric=metric,
            min_threshold=threshold,
            **kwargs,
        )
    expected = {}
    for items in itemsets["itemsets"]:
        for size in range(1, len(items)):
            for selected in combinations(items, size):
                antecedent = frozenset(selected)
                consequent = items - antecedent
                metrics = _rule_metrics(data, antecedent, consequent)
                if metrics[metric] >= threshold:
                    expected[antecedent, consequent] = metrics
    assert set(zip(result.antecedents, result.consequents)) == set(expected)
    for _, row in result.iterrows():
        metrics = expected[row.antecedents, row.consequents]
        for key in result.columns[2:]:
            np.testing.assert_allclose(row[key], metrics[key], atol=1e-14)
    pd.testing.assert_frame_equal(data, original_data)
    pd.testing.assert_frame_equal(itemsets, original_itemsets)
    return result


def test_nullable_counts_match_all_three_state_two_item_tables():
    # A complete all-present transaction keeps every reference denominator
    # positive, while the other three rows cover every 0/1/missing pattern.
    for flat in product((0.0, 1.0, np.nan), repeat=6):
        data = pd.DataFrame(
            np.vstack([np.asarray(flat).reshape(3, 2), [1.0, 1.0]]),
            columns=["A", "B"],
        )
        _check_rules(data)


@pytest.mark.parametrize("labels", [["A", "B", "C"], [0, 1, 2]])
@pytest.mark.parametrize("index", [None, [7, 7, 2, 9, 2, 8, 1, 9]])
def test_multi_item_rules_keep_positional_missingness(labels, index):
    data = pd.DataFrame(
        [
            [1, 1, 1],
            [1, np.nan, 1],
            [np.nan, 1, 1],
            [1, 1, np.nan],
            [0, np.nan, 1],
            [1, 0, 0],
            [np.nan, np.nan, np.nan],
            [0, 0, 0],
        ],
        columns=labels,
        index=index,
    )
    assert len(_check_rules(data)) == 12


@pytest.mark.parametrize("metric", ["confidence", "representativity", "kulczynski"])
@pytest.mark.parametrize("threshold", [0.4, 0.75, 1.0, 1.1])
def test_nullable_counts_filter_the_same_rules(metric, threshold):
    data = pd.DataFrame(
        [[1, 1, 1], [1, np.nan, 0], [0, 1, 1], [np.nan, 1, np.nan]],
        columns=["A", "B", "C"],
    )
    metrics = ["support", "confidence", "representativity", "kulczynski"]
    result = _check_rules(data, metric, threshold, metrics)
    assert list(result.columns) == ["antecedents", "consequents"] + metrics


def test_nullable_counts_without_missing_values_match_complete_mode():
    data = pd.DataFrame(
        [[1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]], columns=["A", "B", "C"]
    )
    nullable = _check_rules(data)
    complete = association_rules(_supports(data), metric="support", min_threshold=0)
    pd.testing.assert_frame_equal(nullable, complete)


def test_nullable_support_only_keeps_incomplete_itemset_input():
    data = pd.DataFrame([[1, 1], [1, np.nan], [np.nan, 0]], columns=["A", "B"])
    itemsets = _supports(data).iloc[[-1]]
    result = association_rules(
        itemsets, len(data), data, null_values=True, support_only=True, min_threshold=0
    )
    assert len(result) == 2
    np.testing.assert_array_equal(result["support"], [1, 1])
    assert (
        result.drop(columns=["antecedents", "consequents", "support"])
        .isna()
        .all()
        .all()
    )
