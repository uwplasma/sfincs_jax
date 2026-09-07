# DKX

[![PyPI](https://img.shields.io/pypi/v/dkx)](https://pypi.org/project/dkx/)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/DKX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/DKX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/sfincs-jax?label=docs)](https://sfincs-jax.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/uwplasma/DKX)](LICENSE)

**Neoclassical transport for stellarators and tokamaks, in JAX.**

DKX solves radially local drift-kinetic equations to calculate particle and heat
fluxes, parallel flows, bootstrap current, transport coefficients and stellarator
ambipolar electric fields. Use physical-unit case files for profiles, or compose
prepared solves and checked implicit derivatives in Python.

- **Physics:** pitch-angle scattering and linearized Fokker–Planck collisions;
  analytic, VMEC and Boozer geometry; SFINCS-v3 input/output compatibility.
- **Repeated calculations:** prepared profile/field updates, Krylov warm starts
  and recycling, shared factors and memory-budgeted independent batches.
- **CPU/GPU and derivatives:** JIT-compatible expert solves and batch sharding;
  implicit forward/transpose checks on supported full-state derivative paths.

Native Case execution supports a restricted DKES/no-Phi1 domain. Richer trajectory,
Phi1 and transport-matrix functionality uses the compatibility/expert paths.
[Capabilities](docs/capabilities.rst) and [derivative scope](docs/differentiability.rst)
describe supported combinations and fixed dependencies.

![Example W7-X geometry, parallel current, bootstrap profile and ambipolar electric field from DKX](docs/_static/figures/readme/w7x_showcase.png)

*Example W7-X output. Resolution admission is observable-specific: the recorded
particle/heat-flux checks do not establish a converged bootstrap-current profile.
See [validation](docs/validation_matrix.rst).*

## Install and run

Python 3.11 or newer:

```bash
python -m pip install dkx             # released CPU package
python -m pip install -U "jax[cuda12]" # optional NVIDIA backend
```

For the examples and development APIs shown here, use this source checkout:

```bash
git clone https://github.com/uwplasma/DKX.git
cd DKX
python -m pip install -e .
dkx run examples/01_tokamak_profile/case.toml --out result.nc
dkx inspect result.nc
dkx converge examples/01_tokamak_profile/case.toml
```

The first case is a small **teaching grid**, not converged transport data.
`dkx converge` checks separate and joint refinements; check every observable you
intend to use. See [installation](docs/installation.rst) for backend setup.

The same case runs from Python:

```python
import dkx

case = dkx.Case.from_file("examples/01_tokamak_profile/case.toml")
result = dkx.run(case)
print(result.arrays["particle_flux_m2_s"])
print(result.metadata["solver_route"])
```

Replace the geometry and species profiles in a [TOML or JSON case](docs/case_files.rst).
`dkx validate case.toml` checks supported inputs. `dkx schema --format toml`
prints a field reference that needs editing before execution.

## Choose a workflow

Run example scripts from the repository root. They save results and figures;
edit their physical and numerical parameters for your application.

| Calculation | Entry point |
| --- | --- |
| Tokamak profile with prescribed Er | `python examples/01_tokamak_profile/run.py` |
| Stellarator from VMEC / Boozer | `python examples/02_vmec_stellarator/run.py` / `python examples/03_boozer_stellarator/run.py` |
| Monoenergetic transport scan | `python examples/04_monoenergetic_scan/run.py` |
| Stellarator ambipolar roots and selection | `python examples/05_ambipolar_profile/run.py` |
| Resolution study / gradient checks | `python examples/06_convergence_certificate/run.py` / `python examples/07_gradients/run.py` |
| Geometry sensitivity and descent | `python examples/08_vmex_optimization/run.py` — analytic geometry proxy |
| Phi1 and impurities | `python examples/09_phi1_and_impurities/run.py` — expert physics example |

[Example guide](docs/examples.rst) · [Real VMEX integration scope](docs/vmex_workflow.rst)

For repeated field scans and gradients, prepare once:

```python
import jax
import jax.numpy as jnp

problem = dkx.prepare_er_scan(case, surface_index=1)

def total_current(er_kv_m):
    scan = dkx.batched_er_scan(
        problem, er_kv_m, devices="auto", differentiable=True,
        retain_full_state=True, max_batch=1,
    )
    return jnp.sum(scan.moments["FSABjHat"])

value, gradient = jax.jit(jax.value_and_grad(total_current))(
    jnp.array([-0.2, 0.0, 0.2])
)
```

Here Er is in kV/m and `FSABjHat` is normalized parallel current. Preparation fixes
geometry/profiles; opt-in profile updates and reuse limits are documented in
[differentiability](docs/differentiability.rst). `devices="auto"` shards independent
inputs across available local devices; each linear system stays on one device.
See [parallelism](docs/parallelism.rst) for memory and per-input convergence checks.
Root derivatives require a regular retained branch; finite sampling cannot prove
that every root has been found.

SFINCS decks and outputs remain supported:

```bash
dkx input.namelist --out sfincsOutput.h5
dkx --plot sfincsOutput.h5
```

## Demonstrated results

- Matched SFINCS-v3 equations have field-by-field regression checks. Independent
  zero-field monoenergetic comparisons with YANCC/MONKES pass the recorded 6%
  coefficient gate on three configurations. This does not validate full-FP,
  finite-Er or ambipolar physics: [evidence and scope](validation/README.md).
- On a 7,850-unknown, three-field PAS objective, retained Schur factors reduced
  warm value/gradient medians **67.59 → 37.09 ms on CPU** and **221.74 → 195.56 ms
  on A4000**, versus checked recomputation. Twelve paired runs on each host;
  these are development measurements, not complete optimization timings or a
  CPU/GPU ranking: [measurement context](plan.md#useful-results-with-their-limits).

[Performance](docs/performance.rst) · [Physics](docs/physics_models.rst) ·
[API](docs/api.rst) · [Research plan](plan.md) · [Contributing](docs/contributing.rst)

DKX owns physics and solver policy; [SOLVAX](https://github.com/uwplasma/SOLVAX)
owns reusable solver algorithms. See [LICENSE](LICENSE).
