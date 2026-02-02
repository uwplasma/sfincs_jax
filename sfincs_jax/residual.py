from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .v3_fblock import V3FBlockOperator, matvec_v3_fblock_flat
from .v3_system import V3FullSystemOperator, apply_v3_full_system_operator


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3FBlockLinearSystem:
    """Linear system for the v3 distribution-function block (BLOCK_F).

    The residual is:

      r(x) = A x - b

    where A is represented matrix-free by :class:`sfincs_jax.v3_fblock.V3FBlockOperator`.
    """

    op: V3FBlockOperator
    b_flat: jnp.ndarray  # (op.flat_size,)

    def tree_flatten(self):
        children = (self.op, self.b_flat)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        op, b_flat = children
        return cls(op=op, b_flat=b_flat)

    def residual(self, x_flat: jnp.ndarray) -> jnp.ndarray:
        """Compute r(x) = A x - b."""
        x_flat = jnp.asarray(x_flat)
        return matvec_v3_fblock_flat(self.op, x_flat) - self.b_flat

    def jacobian_matvec(self, v_flat: jnp.ndarray) -> jnp.ndarray:
        """Compute (dr/dx) v, matrix-free.

        For linear problems this is just A v. We keep this method explicit to make it easy
        to swap in a nonlinear residual later while keeping a matrix-free interface.
        """
        v_flat = jnp.asarray(v_flat)
        return matvec_v3_fblock_flat(self.op, v_flat)

    def jvp(self, x_flat: jnp.ndarray, v_flat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (r(x), (dr/dx) v) using JAX's JVP."""

        def f(x):
            return self.residual(x)

        return jax.jvp(f, (jnp.asarray(x_flat),), (jnp.asarray(v_flat),))


residual_v3_fblock_jit = jax.jit(lambda sys, x: sys.residual(x), static_argnums=())
jacobian_matvec_v3_fblock_jit = jax.jit(lambda sys, v: sys.jacobian_matvec(v), static_argnums=())


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3FullLinearSystem:
    """Linear system for the v3 full operator currently supported by :class:`sfincs_jax.v3_system.V3FullSystemOperator`.

    The residual is:

      r(x) = A x - b

    where A is represented matrix-free by :class:`sfincs_jax.v3_system.V3FullSystemOperator`.
    """

    op: V3FullSystemOperator
    b_full: jnp.ndarray  # (op.total_size,)

    def tree_flatten(self):
        children = (self.op, self.b_full)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        op, b_full = children
        return cls(op=op, b_full=b_full)

    def residual(self, x_full: jnp.ndarray) -> jnp.ndarray:
        """Compute r(x) = A x - b."""
        x_full = jnp.asarray(x_full)
        return apply_v3_full_system_operator(self.op, x_full) - self.b_full

    def jacobian_matvec(self, v_full: jnp.ndarray) -> jnp.ndarray:
        """Compute (dr/dx) v, matrix-free."""
        v_full = jnp.asarray(v_full)
        return apply_v3_full_system_operator(self.op, v_full)

    def jvp(self, x_full: jnp.ndarray, v_full: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (r(x), (dr/dx) v) using JAX's JVP."""

        def f(x):
            return self.residual(x)

        return jax.jvp(f, (jnp.asarray(x_full),), (jnp.asarray(v_full),))


residual_v3_full_system_jit = jax.jit(lambda sys, x: sys.residual(x), static_argnums=())
jacobian_matvec_v3_full_system_jit = jax.jit(lambda sys, v: sys.jacobian_matvec(v), static_argnums=())
