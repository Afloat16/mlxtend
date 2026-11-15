from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mlxtend.frequent_patterns import fpgrowth, fpmax, hmine


def _transactions(case):
    if case == "quarter-hundred":
        return (
            pd.DataFrame(
                [[True, True, False]] * 7 + [[False, False, True]] * 18,
                columns=["A", "B", "C"],
            ),
            7 / 25,
        )
    if case == "hundred":
        return (
            pd.DataFrame(
                [[True, True, False]] * 7 + [[False, False, True]] * 93,
                columns=["A", "B", "C"],
            ),
            7 / 100,
        )
    return (
        pd.DataFrame([[True, True], [True, False], [False, True]], columns=["A", "B"]),
        1 / 3,
    )


def _enumerated_itemsets(transactions, threshold, maximal):
    values = transactions.to_numpy()
    expected = {}
    for length in range(1, len(transactions.columns) + 1):
        for indices in combinations(range(len(transactions.columns)), length):
            count = np.count_nonzero(values[:, indices].all(axis=1))
            support = np.float64(count) / len(transactions)
            if support >= threshold:
                items = frozenset(transactions.columns[list(indices)])
                expected[items] = support
    if maximal:
        expected = {
            items: support
            for items, support in expected.items()
            if not any(items < other for other in expected)
        }
    return expected


@pytest.mark.parametrize("miner", [fpgrowth, fpmax, hmine])
@pytest.mark.parametrize("case", ["quarter-hundred", "hundred", "third"])
@pytest.mark.parametrize("direction", [-1, 0, 1])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("threshold_type", [np.float64, np.float32])
def test_support_counts_match_fractional_thresholds(
    miner, case, direction, sparse, threshold_type
):
    transactions, boundary = _transactions(case)
    threshold = threshold_type(boundary)
    if direction:
        target = threshold_type(-np.inf if direction < 0 else np.inf)
        threshold = np.nextafter(threshold, target)
    expected = _enumerated_itemsets(transactions, threshold, miner is fpmax)
    if sparse:
        transactions = transactions.astype(pd.SparseDtype(bool, False))
    original = transactions.copy(deep=True)

    result = miner(transactions, min_support=threshold, use_colnames=True)

    actual = dict(zip(result["itemsets"], result["support"].astype(float)))
    assert actual == expected
    pd.testing.assert_frame_equal(transactions, original)
