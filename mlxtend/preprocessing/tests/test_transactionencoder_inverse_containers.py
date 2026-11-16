# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from mlxtend.preprocessing import TransactionEncoder

TRANSACTIONS = [["pear", "apple", "pear"], [], ["banana"], ["pear", "banana"]]
EXPECTED = [["apple", "pear"], [], ["banana"], ["banana", "pear"]]


@pytest.mark.parametrize(
    "container", [np.asarray, csr_matrix, csc_matrix, csr_array, csc_array]
)
def test_inverse_round_trip_array_containers(container):
    encoder = TransactionEncoder().fit(TRANSACTIONS)
    encoded = container(encoder.transform(TRANSACTIONS))
    assert encoder.inverse_transform(encoded) == EXPECTED


def test_inverse_round_trip_encoder_sparse_output():
    encoder = TransactionEncoder().fit(TRANSACTIONS)
    encoded = encoder.transform(TRANSACTIONS, sparse=True)
    assert encoder.inverse_transform(encoded) == EXPECTED


@pytest.mark.parametrize("reorder", [False, True])
def test_inverse_round_trip_pandas_output(reorder):
    encoder = TransactionEncoder().set_output(transform="pandas")
    encoded = encoder.fit_transform(TRANSACTIONS)
    encoded.index = [8, 3, 3, 1]
    if reorder:
        encoded = encoded[["pear", "apple", "banana"]]
    original = encoded.copy(deep=True)

    assert encoder.inverse_transform(encoded) == EXPECTED
    pd.testing.assert_frame_equal(encoded, original)


@pytest.mark.parametrize(
    "labels",
    [
        ["apple", "apple", "pear"],
        ["apple", "banana"],
        ["apple", "banana", "unknown"],
        ["apple", "banana", "pear", "unknown"],
    ],
)
def test_inverse_rejects_ambiguous_dataframe_columns(labels):
    encoder = TransactionEncoder().fit(TRANSACTIONS)
    encoded = pd.DataFrame(np.ones((2, len(labels)), dtype=bool), columns=labels)
    with pytest.raises(ValueError, match="columns must match the fitted items"):
        encoder.inverse_transform(encoded)


def test_inverse_dataframe_numeric_item_labels():
    transactions = [[20, 10], [], [30]]
    encoder = TransactionEncoder().fit(transactions)
    encoded = pd.DataFrame(encoder.transform(transactions), columns=encoder.columns_)
    assert encoder.inverse_transform(encoded[[30, 10, 20]]) == [[10, 20], [], [30]]


@pytest.mark.parametrize(
    "container", [np.asarray, csr_matrix, csc_matrix, csr_array, pd.DataFrame]
)
@pytest.mark.parametrize("shape", [(0, 3), (2, 3), (0, 0), (2, 0)])
def test_inverse_empty_transactions_and_empty_matrices(container, shape):
    items = ["apple", "banana", "pear"][: shape[1]]
    encoder = TransactionEncoder().fit([items])
    encoded = container(np.zeros(shape, dtype=bool))
    if isinstance(encoded, pd.DataFrame):
        encoded.columns = items
    assert encoder.inverse_transform(encoded) == [[] for _ in range(shape[0])]


@pytest.mark.parametrize("transform_output", ["default", "pandas"])
@pytest.mark.parametrize("transactions", [[], [[], []]])
def test_inverse_dense_empty_batch_round_trip(transform_output, transactions):
    encoder = TransactionEncoder().set_output(transform=transform_output)
    encoder.fit(TRANSACTIONS)
    assert encoder.inverse_transform(encoder.transform(transactions)) == transactions


def test_inverse_sparse_does_not_densify_or_modify_input(monkeypatch):
    encoder = TransactionEncoder().fit([["apple", "banana", "pear"]])
    # Unsorted duplicate coordinates and explicit zeros must agree with the
    # logical values, without changing the caller's sparse buffers or flags.
    encoded = csr_matrix(
        (np.array([1, 1, 0, 1, -1, 1]), [2, 0, 1, 1, 1, 2], [0, 3, 6, 6]),
        shape=(3, 3),
    )
    original = (encoded.data.copy(), encoded.indices.copy(), encoded.indptr.copy())
    flags = (encoded.has_sorted_indices, encoded.has_canonical_format)

    def reject_dense(*args, **kwargs):
        raise AssertionError("inverse_transform must not densify sparse input")

    monkeypatch.setattr(csr_matrix, "toarray", reject_dense)
    monkeypatch.setattr(csr_matrix, "todense", reject_dense)
    assert encoder.inverse_transform(encoded) == [["apple", "pear"], ["pear"], []]
    for actual, expected in zip(
        (encoded.data, encoded.indices, encoded.indptr), original
    ):
        np.testing.assert_array_equal(actual, expected)
    assert (encoded.has_sorted_indices, encoded.has_canonical_format) == flags


def test_inverse_sparse_dataframe_without_densifying(monkeypatch):
    encoder = TransactionEncoder().fit(TRANSACTIONS)
    encoded = pd.DataFrame.sparse.from_spmatrix(
        encoder.transform(TRANSACTIONS, sparse=True), columns=encoder.columns_
    )[["pear", "banana", "apple"]]
    original = encoded.copy(deep=True)

    def reject_dense(*args, **kwargs):
        raise AssertionError("zero-filled sparse DataFrames should stay sparse")

    monkeypatch.setattr(pd.DataFrame, "to_numpy", reject_dense)
    assert encoder.inverse_transform(encoded) == EXPECTED
    pd.testing.assert_frame_equal(encoded, original)


def test_inverse_sparse_dataframe_with_true_fill_value():
    encoder = TransactionEncoder().fit(TRANSACTIONS)
    encoded = pd.DataFrame(
        encoder.transform(TRANSACTIONS),
        columns=encoder.columns_,
        dtype=pd.SparseDtype(bool, True),
    )
    assert encoder.inverse_transform(encoded) == EXPECTED


def test_inverse_dense_list_input_unchanged():
    encoder = TransactionEncoder().fit(TRANSACTIONS)
    encoded = encoder.transform(TRANSACTIONS).tolist()
    original = [row[:] for row in encoded]
    assert encoder.inverse_transform(encoded) == EXPECTED
    assert encoded == original
