"""H-Mine must identify input features by position before naming its output."""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mlxtend.frequent_patterns import hmine


def brute_force_itemsets(values, columns, min_support, use_colnames):
    expected = []
    for size in range(1, values.shape[1] + 1):
        for indices in combinations(range(values.shape[1]), size):
            support = values[:, indices].all(axis=1).mean()
            if support >= min_support:
                items = [columns[i] for i in indices] if use_colnames else indices
                expected.append((frozenset(items), support))
    return Counter(expected)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("use_colnames", [False, True])
@pytest.mark.parametrize(
    "values,columns",
    [
        ([[1, 0], [1, 1]], ["item", "item"]),
        (
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]],
            ["rare", "item", "item"],
        ),
        ([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]], ["item"] * 3),
        ([[1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 1, 1]], [0, 2, 4]),
    ],
)
def test_itemset_supports_match_positional_enumeration(
    values, columns, sparse, use_colnames
):
    values = np.asarray(values, dtype=bool)
    frame = pd.DataFrame(values, columns=columns)
    if sparse:
        frame = frame.astype(pd.SparseDtype(bool, False))
    actual = hmine(frame, min_support=0.5, use_colnames=use_colnames)
    observed = Counter(
        (row.itemsets, row.support) for row in actual.itertuples(index=False)
    )
    assert observed == brute_force_itemsets(values, columns, 0.5, use_colnames)


@pytest.mark.parametrize("use_colnames", [False, True])
def test_maximum_length_does_not_merge_duplicate_singletons(use_colnames):
    values = np.asarray([[1, 0], [1, 1]], dtype=bool)
    columns = ["item", "item"]
    actual = hmine(
        pd.DataFrame(values, columns=columns),
        min_support=0.5,
        max_len=1,
        use_colnames=use_colnames,
    )
    expected = Counter(
        (frozenset([columns[i] if use_colnames else i]), values[:, i].mean())
        for i in range(values.shape[1])
    )
    assert (
        Counter((row.itemsets, row.support) for row in actual.itertuples(index=False))
        == expected
    )


def test_verbose_output_keeps_input_column_names(capsys):
    frame = pd.DataFrame([[True, True], [True, False]], columns=["alpha", "beta"])
    hmine(frame, min_support=0.5, verbose=1)
    output = capsys.readouterr().out
    assert "alpha" in output
    assert "beta" in output
