import numpy as np
import pandas as pd
import pytest

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

LABEL_PAIRS = [
    pytest.param((np.str_("milk"), np.str_("bread")), id="numpy-strings"),
    pytest.param((np.bytes_(b"milk"), np.bytes_(b"bread")), id="numpy-bytes"),
    pytest.param((np.float64(1.25), np.float64(1.75)), id="colliding-float64"),
    pytest.param((np.float32(2.25), np.float32(3.75)), id="float32"),
    pytest.param((np.int64(10), np.int64(20)), id="numpy-integers"),
    pytest.param((np.bool_(False), np.bool_(True)), id="numpy-booleans"),
    pytest.param(("milk", "bread"), id="python-strings"),
    pytest.param(((1, "milk"), (2, "bread")), id="tuple-labels"),
]


def _itemsets(labels):
    left, right = labels
    return pd.DataFrame(
        {
            "support": [0.75, 0.5, 0.25],
            "itemsets": [
                frozenset([left]),
                frozenset([right]),
                frozenset([left, right]),
            ],
        }
    )


def _normalize_rules(rules, labels):
    label_indices = {label: index for index, label in enumerate(labels)}
    result = rules.copy()
    for column in ["antecedents", "consequents"]:
        result[column] = result[column].map(
            lambda items: frozenset(label_indices[item] for item in items)
        )
    order = result["antecedents"].map(lambda items: min(items)).to_numpy().argsort()
    return result.iloc[order].reset_index(drop=True)


@pytest.mark.parametrize("labels", LABEL_PAIRS)
@pytest.mark.parametrize("support_only", [False, True])
def test_numpy_item_labels_preserve_rules_and_metrics(labels, support_only):
    itemsets = _itemsets(labels)
    original = itemsets.copy(deep=True)

    result = association_rules(itemsets, min_threshold=0, support_only=support_only)
    reference = association_rules(
        _itemsets(("left", "right")), min_threshold=0, support_only=support_only
    )

    assert len(result) == 2
    normalized = _normalize_rules(result, labels)
    pd.testing.assert_frame_equal(
        normalized, _normalize_rules(reference, ("left", "right"))
    )
    np.testing.assert_allclose(normalized["support"], [0.25, 0.25])
    if not support_only:
        np.testing.assert_allclose(normalized["confidence"], [1 / 3, 1 / 2])
        np.testing.assert_allclose(normalized["lift"], [2 / 3, 2 / 3])
    pd.testing.assert_frame_equal(itemsets, original)


@pytest.mark.parametrize("miner", [apriori, fpgrowth])
@pytest.mark.parametrize("labels", LABEL_PAIRS[:5])
def test_mined_numpy_item_labels_preserve_rule_identity(miner, labels):
    transactions = pd.DataFrame(
        [[True, True], [True, False], [True, False], [False, True]],
        columns=pd.Index(list(labels), dtype=object),
    )
    original = transactions.copy(deep=True)
    itemsets = miner(transactions, min_support=0.25, use_colnames=True)

    result = association_rules(itemsets, min_threshold=0)

    assert len(result) == 2
    normalized = _normalize_rules(result, labels)
    reference = _normalize_rules(
        association_rules(_itemsets(("left", "right")), min_threshold=0),
        ("left", "right"),
    )
    pd.testing.assert_frame_equal(normalized, reference)
    pd.testing.assert_frame_equal(transactions, original)
