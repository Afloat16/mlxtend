from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mlxtend.frequent_patterns import fpmax


def exhaustive_maximal(values, columns, min_support, max_len):
    frequent = {}
    for size in range(1, values.shape[1] + 1):
        for indices in combinations(range(values.shape[1]), size):
            support = values[:, indices].all(axis=1).mean()
            if support >= min_support:
                itemset = frozenset(columns[index] for index in indices)
                frequent[itemset] = support
    return {
        itemset: support
        for itemset, support in frequent.items()
        if (max_len is None or len(itemset) <= max_len)
        and not any(itemset < superset for superset in frequent)
    }


@pytest.mark.parametrize("max_len", [1, 2, 3, 4, None])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("use_colnames", [False, True])
def test_length_limit_filters_global_maximal_itemsets(max_len, sparse, use_colnames):
    rng = np.random.RandomState(421)
    datasets = [rng.randint(0, 2, (8, 6)).astype(bool) for _ in range(96)]
    datasets.extend([np.ones((8, 6), dtype=bool), np.zeros((8, 6), dtype=bool)])
    names = ["item-" + str(index) for index in range(6)]
    columns = names if use_colnames else range(6)

    for index, values in enumerate(datasets):
        threshold = [0.25, 0.5, 0.75][index % 3]
        frame = pd.DataFrame(values, columns=names)
        if sparse:
            frame = frame.astype(pd.SparseDtype(bool, False))
        expected = exhaustive_maximal(values, columns, threshold, max_len)
        result = fpmax(
            frame,
            min_support=threshold,
            max_len=max_len,
            use_colnames=use_colnames,
        )
        actual = dict(zip(result["itemsets"], result["support"]))
        assert actual == expected, (index, threshold, max_len, values)
