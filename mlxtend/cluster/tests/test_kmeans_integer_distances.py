import numpy as np
import pytest

from mlxtend.cluster import Kmeans

DTYPES_AND_SPANS = [
    pytest.param(np.int8, 100, id="int8"),
    pytest.param(np.uint8, 200, id="uint8"),
    pytest.param(np.int16, 300, id="int16"),
    pytest.param(np.uint16, 1000, id="uint16"),
    pytest.param(np.int32, 100000, id="int32"),
    pytest.param(np.uint32, 100000, id="uint32"),
    pytest.param(np.int64, 4000000000, id="int64"),
    pytest.param(np.float32, 200, id="float32"),
    pytest.param(np.float64, 200, id="float64"),
]


@pytest.mark.parametrize("dtype,span", DTYPES_AND_SPANS)
def test_integer_prediction_matches_exact_nearest_centroid(dtype, span):
    samples = np.array([[0], [span]], dtype=dtype)
    original = samples.copy()
    model = Kmeans(k=2, max_iter=10, random_seed=0).fit(samples)
    queries = np.array(
        [[span // 8], [3 * span // 8], [5 * span // 8], [7 * span // 8]],
        dtype=dtype,
    )
    original_queries = queries.copy()

    assert np.isfinite(model.centroids_).all()
    np.testing.assert_allclose(np.sort(model.centroids_[:, 0]), [0, span])

    # Python integers give an independent, overflow-free distance reference.
    expected = []
    for query in queries:
        distances = [
            sum((int(value) - int(center)) ** 2 for value, center in zip(query, c))
            for c in model.centroids_
        ]
        expected.append(distances.index(min(distances)))

    np.testing.assert_array_equal(model.predict(queries), expected)
    np.testing.assert_array_equal(samples, original)
    np.testing.assert_array_equal(queries, original_queries)


@pytest.mark.parametrize("dtype,span", DTYPES_AND_SPANS)
def test_integer_lloyd_step_matches_floating_input(dtype, span):
    samples = np.array(
        [[0], [span // 100], [span - span // 100], [span]], dtype=dtype
    )
    original = samples.copy()
    expected = Kmeans(k=2, max_iter=1, random_seed=0).fit(samples.astype(np.float64))

    result = Kmeans(k=2, max_iter=1, random_seed=0).fit(samples)

    assert np.isfinite(result.centroids_).all()
    np.testing.assert_allclose(result.centroids_, expected.centroids_)
    assert result.clusters_ == expected.clusters_
    np.testing.assert_array_equal(samples, original)
