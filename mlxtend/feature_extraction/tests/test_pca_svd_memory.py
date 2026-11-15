import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mlxtend.feature_extraction import PrincipalComponentAnalysis


@pytest.mark.parametrize("shape", [(15, 3), (4, 9), (6, 6)])
@pytest.mark.parametrize("n_components", [None, 1, 3])
def test_svd_retains_full_feature_basis(shape, n_components):
    samples = np.random.RandomState(17).normal(size=shape)
    original = samples.copy()

    model = PrincipalComponentAnalysis(n_components=n_components).fit(samples)

    covariance = np.cov(samples, rowvar=False)
    expected_values = np.linalg.eigvalsh(covariance)[::-1]
    n_features = shape[1]
    assert model.e_vecs_.shape == (n_features, n_features)
    assert model.e_vals_.shape == (n_features,)
    np.testing.assert_allclose(model.e_vals_, expected_values, atol=1e-12)
    np.testing.assert_allclose(
        model.e_vecs_.T @ model.e_vecs_, np.eye(n_features), atol=1e-12
    )
    np.testing.assert_allclose(
        model.e_vecs_ @ np.diag(model.e_vals_) @ model.e_vecs_.T,
        covariance,
        atol=1e-12,
    )
    assert model.transform(samples).shape == (
        shape[0],
        n_features if n_components is None else n_components,
    )
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("n_components", [1, 3])
def test_reduced_svd_preserves_whitening(n_components):
    samples = np.random.RandomState(23).normal(size=(24, 3))
    model = PrincipalComponentAnalysis(n_components=n_components, whitening=True).fit(
        samples
    )

    transformed = model.transform(samples)

    np.testing.assert_allclose(
        np.atleast_2d(np.cov(transformed, rowvar=False)),
        np.eye(n_components),
        atol=1e-12,
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="The allocation regression uses Linux virtual-memory limits",
)
@pytest.mark.parametrize("n_components", [None, 1])
def test_tall_data_fits_without_square_sample_allocation(n_components):
    script = textwrap.dedent(f"""
        import os
        from pathlib import Path
        import resource

        import numpy as np
        from mlxtend.feature_extraction import PrincipalComponentAnalysis

        # Set the limit after imports, allowing another 256 MiB for data,
        # decomposition and predictions. A 12000-by-12000 float64 basis alone
        # would need over 1 GiB.
        current = int(Path("/proc/self/statm").read_text().split()[0])
        current *= os.sysconf("SC_PAGE_SIZE")
        limit = current + 256 * 1024**2
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        samples = np.random.RandomState(11).normal(size=(12000, 3))
        model = PrincipalComponentAnalysis(n_components={n_components!r}).fit(samples)
        result = model.transform(samples[:20])
        assert result.shape == (20, {3 if n_components is None else n_components})
        assert np.isfinite(result).all()
        print("fit and transform succeeded with 256 MiB additional address space")
        """)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    run = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )

    assert run.returncode == 0, run.stdout + run.stderr
