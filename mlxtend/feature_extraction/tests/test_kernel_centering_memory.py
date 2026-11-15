import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from mlxtend.feature_extraction import RBFKernelPCA


@pytest.mark.parametrize("kind", ["regular", "duplicates", "constant"])
@pytest.mark.parametrize("gamma", [0.1, 2.0])
@pytest.mark.parametrize("copy_X", [False, True])
def test_fitted_kernel_matches_dense_centering_reference(kind, gamma, copy_X):
    samples = np.random.RandomState(19).normal(size=(18, 3))
    if kind == "duplicates":
        samples[9:] = samples[:9]
    elif kind == "constant":
        samples[:] = 3
    before = samples.copy()
    squared_distances = ((samples[:, None] - samples[None, :]) ** 2).sum(axis=2)
    kernel = np.exp(-gamma * squared_distances)
    center = np.eye(len(samples)) - np.ones((len(samples), len(samples))) / len(samples)
    reference = center @ kernel @ center

    model = RBFKernelPCA(gamma=gamma, n_components=2, copy_X=copy_X).fit(samples)

    np.testing.assert_allclose(
        model.e_vals_, np.linalg.eigvalsh(reference)[::-1], rtol=1e-12, atol=1e-12
    )
    reconstructed = (model.e_vecs_ * model.e_vals_) @ model.e_vecs_.T
    np.testing.assert_allclose(reconstructed, reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(reconstructed.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(reconstructed.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(
        model.X_projected_.T @ model.X_projected_, np.eye(2), atol=1e-12
    )
    np.testing.assert_array_equal(samples, before)
    assert np.shares_memory(model.X_, samples) == (not copy_X)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="The allocation regression uses Linux virtual-memory limits",
)
def test_full_fit_avoids_dense_centering_matrix_allocations():
    script = textwrap.dedent("""
        import os
        from pathlib import Path
        import resource

        import numpy as np

        from mlxtend.feature_extraction import RBFKernelPCA

        current = int(Path("/proc/self/statm").read_text().split()[0])
        current *= os.sysconf("SC_PAGE_SIZE")
        limit = current + 700 * 1024**2
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))

        samples = np.random.RandomState(7).normal(size=(4000, 3))
        model = RBFKernelPCA(gamma=0.5, n_components=2).fit(samples)
        result = model.transform(samples[:6])
        assert model.X_projected_.shape == (4000, 2)
        assert result.shape == (6, 2)
        assert np.isfinite(model.e_vals_).all()
        assert np.isfinite(result).all()
        print("fit and transform succeeded with 700 MiB additional address space")
        """)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    run = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )

    assert run.returncode == 0, run.stdout + run.stderr
