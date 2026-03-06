from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sfincs_jax.v3_driver import _build_sparse_ilu_from_matvec


def test_chunked_sparse_assembly_matches_dense_operator(monkeypatch) -> None:
    a = jnp.array(
        [
            [4.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 4.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 4.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 4.0],
        ],
        dtype=jnp.float64,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK_MIN", "1")

    a_csr_full, a_csr_drop, ilu, a_dense, l_dense, u_dense, l_unit_diag = _build_sparse_ilu_from_matvec(
        matvec=lambda x: a @ x,
        n=6,
        dtype=jnp.float64,
        cache_key=("chunked_sparse_assembly_test", 6),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=1.0e-4,
        fill_factor=10.0,
        build_dense_factors=False,
        build_jax_factors=False,
        build_ilu=False,
        store_dense=False,
        emit=None,
    )

    a_np = np.asarray(a)
    np.testing.assert_allclose(np.asarray(a_csr_full.toarray()), a_np, rtol=0.0, atol=0.0)
    drop_np = np.asarray(a_csr_drop.toarray())
    np.testing.assert_allclose(np.diag(drop_np) - np.diag(a_np), np.full((6,), 4.0e-12), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(drop_np - np.diag(np.diag(drop_np)), a_np - np.diag(np.diag(a_np)), rtol=0.0, atol=0.0)
    assert ilu is None
    assert a_dense is None
    assert l_dense is None
    assert u_dense is None
    assert l_unit_diag is True
