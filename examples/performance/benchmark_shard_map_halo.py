from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.v3_fblock import fblock_operator_from_namelist
from sfincs_jax.collisionless import apply_collisionless_v3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate shard_map + explicit halo (all_gather) for collisionless d/dtheta."
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "examples" / "performance" / "transport_parallel_sharded.input.namelist"

    parser.add_argument("--input", type=Path, default=default_input, help="input.namelist path")
    parser.add_argument("--devices", type=int, default=4, help="CPU devices to use")
    parser.add_argument("--axis", choices=("theta", "zeta"), default="theta", help="Shard axis")
    parser.add_argument("--repeats", type=int, default=5, help="repeats for timing")
    args = parser.parse_args()

    os.environ["SFINCS_JAX_CPU_DEVICES"] = str(int(args.devices))
    os.environ["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    os.environ["SFINCS_JAX_SOLVER_ITER_STATS"] = "0"

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec
    from jax import shard_map

    nml = read_sfincs_input(args.input)
    fblock = fblock_operator_from_namelist(nml=nml)
    rng = np.random.default_rng(0)
    f = rng.normal(size=fblock.f_shape).astype(np.float64)
    f = jnp.asarray(f)

    axis_name = args.axis
    mesh = Mesh(np.array(jax.local_devices()), (axis_name,))

    if axis_name == "theta":
        in_spec = PartitionSpec(None, None, None, axis_name, None)
        out_spec = PartitionSpec(None, None, None, axis_name, None)
        gather_axis = 3
    else:
        in_spec = PartitionSpec(None, None, None, None, axis_name)
        out_spec = PartitionSpec(None, None, None, None, axis_name)
        gather_axis = 4

    def _sharded(f_local: jnp.ndarray) -> jnp.ndarray:
        # Explicit halo exchange via all_gather (full gather for evaluation).
        f_gather = jax.lax.all_gather(f_local, axis_name)
        # Combine shards back into a full theta/zeta dimension.
        f_full = jnp.concatenate(list(f_gather), axis=gather_axis)
        out_full = apply_collisionless_v3(fblock.collisionless, f_full)
        # Return only the local slice to match sharding.
        return jax.lax.dynamic_slice_in_dim(
            out_full,
            start_index=jax.lax.axis_index(axis_name) * f_local.shape[gather_axis],
            slice_size=f_local.shape[gather_axis],
            axis=gather_axis,
        )

    sharded_fn = shard_map(
        _sharded,
        mesh=mesh,
        in_specs=in_spec,
        out_specs=out_spec,
    )

    with mesh:
        # Warmup
        y = sharded_fn(f)
        y.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(max(1, int(args.repeats))):
            y = sharded_fn(f)
        y.block_until_ready()
        t1 = time.perf_counter()

    dt = (t1 - t0) / float(max(1, int(args.repeats)))
    print(f"axis={axis_name} devices={args.devices} mean_s={dt:.6f}")


if __name__ == "__main__":
    main()
