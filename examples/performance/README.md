# Performance

Examples that focus on JIT/vectorization performance:
- `benchmark_jit_matvec.py` — benchmark a JIT-compiled matvec.
- `benchmark_transport_l11_vs_fortran.py` — reproduce the 2x2 L11 parity/runtime figure used in the README/docs.
- `profile_transport_compile_runtime_cache.py` — profile transport-solve compile/runtime split with persistent JAX cache.
