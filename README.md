# DKX

[![PyPI](https://img.shields.io/pypi/v/dkx)](https://pypi.org/project/dkx/)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/DKX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/DKX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/sfincs-jax?label=docs)](https://sfincs-jax.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/uwplasma/DKX)](LICENSE)

**Differentiable neoclassical transport for stellarators and tokamaks, in JAX.**

DKX solves the radially local, linearized drift-kinetic equation on a flux surface
and returns particle and heat fluxes, parallel flows, bootstrap current, transport
matrices and ambipolar radial electric fields. It implements the SFINCS Fortran v3
model, including full Fokker–Planck collisions and Phi1, reads and writes SFINCS decks,
runs on CPU or GPU, and every output is differentiable in every input.

![W7-X standard configuration: |B| and parallel current density on the boundary, bootstrap current profile, ambipolar Er against Pablant et al. 2018](docs/_static/figures/readme/w7x_showcase.png)

## Install

```bash
pip install dkx                    # CPU
pip install -U "jax[cuda12]"       # add for NVIDIA GPUs
```

## Run

```python
import dkx

case = dkx.Case.from_mapping({           # analytic tokamak, teaching grid: seconds, not converged
    "schema": 1, "name": "tokamak", "run": {"workflow": "profile", "progress": False},
    "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.16, 0.25, 0.36]},
    "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                 "density_m3": [8.0e19, 7.0e19, 6.0e19], "temperature_keV": [1.0, 0.8, 0.6]}],
    "physics": {"model": "full_local", "collisions": "pitch_angle_scattering", "magnetic_drifts": "dkes", "phi1": "off"},
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}, "solver": {"method": "auto", "relative_tolerance": 1e-8},
})
result = dkx.run(case)
print("solver route:", result.metadata["solver_route"])
print("particle flux:", float(result.arrays["particle_flux_m2_s"][1, 0]))
```

From the shell: `dkx run case.toml`, `dkx converge case.toml` (check the resolution before
trusting a number), `dkx wout_w7x.nc`, or `dkx input.namelist` for a SFINCS deck.

## Gradients

```python
import jax, jax.numpy as jnp, dkx

case = dkx.Case.from_file("examples/05_ambipolar_profile/case.toml")
problem = dkx.prepare_er_scan(case, surface_index=1)          # geometry, grids, collisions once

def bootstrap_current(er_kv_m):
    scan = dkx.batched_er_scan(problem, er_kv_m, differentiable=True, retain_full_state=True)
    return jnp.sum(scan.moments["FSABjHat"])

j, dj_der = jax.jit(jax.value_and_grad(bootstrap_current))(jnp.array([-0.2, 0.0, 0.2]))
```

The derivative passes through the linear solve by the implicit function theorem, and
every returned state has satisfied the original kinetic equation. Profiles and geometry
differentiate the same way: [differentiability](docs/differentiability.rst).

## Choose a workflow

| Calculation | Start from |
| --- | --- |
| Tokamak profile with prescribed `E_r` | `examples/01_tokamak_profile` |
| Stellarator from VMEC or Boozer files | `examples/02_vmec_stellarator`, `examples/03_boozer_stellarator` |
| Monoenergetic transport scan | `examples/04_monoenergetic_scan` |
| Ambipolar roots and branch selection | `examples/05_ambipolar_profile` |
| Resolution study, gradient checks | `examples/06_convergence_certificate`, `examples/07_gradients` |
| Geometry sensitivity and descent (analytic proxy) | `examples/08_vmex_optimization` |
| Phi1 and impurities (expert path) | `examples/09_phi1_and_impurities` |

## Why DKX

| | DKX | SFINCS v3 | MONKES | yancc |
| --- | :---: | :---: | :---: | :---: |
| Full linearized Fokker–Planck, multispecies | ✅ | ✅ | ❌ | ✅ |
| Analytic, VMEC, Boozer and `lasym` geometry | ✅ | ✅ | ✅ | ✅ |
| Phi1 quasineutrality; Tangential magnetic drifts; `export_f` | ✅ | ✅ | ❌ | ❌ |
| Ambipolar `E_r` root with retained branch evidence | ✅ | ✅ | ❌ | ❌ |
| Transport matrices (RHSMode 2/3) and SFINCS deck/HDF5 I/O | ✅ | ✅ | ❌ | ❌ |
| GPU, JIT-compiled scans, Krylov recycling | ✅ | ❌ | ❌ | ✅ |
| Exact gradients of any output w.r.t. any input | ✅ | adjoint branches | ❌ | claimed |
| Gradients verified against finite differences | ✅ | | | |

## Verified

![DKX against SFINCS Fortran v3, MONKES and YANCC: scaled differences on matched full Fokker-Planck decks, and Beidler-normalized monoenergetic coefficients](docs/_static/figures/readme/cross_code_validation.png)

Against SFINCS Fortran v3 on 38 upstream decks with the same discretization, DKX
agrees to solver tolerance: median 4e-6, full Fokker–Planck decks to 1e-8. Against the
independent codes MONKES and YANCC, the four Beidler-normalized monoenergetic coefficients
agree within 6 percent and `D33` within 0.1 percent on three configurations. Gradients
agree with central finite differences over a step window on every shipped derivative
example. Details, tolerances and scope: [validation matrix](docs/validation_matrix.rst).

## Fast

![Runtime and peak memory, DKX against SFINCS Fortran v3, on the 744k-unknown HSX PAS case](docs/_static/figures/readme/tier1_hsx_runtime_memory.png)

`HSX_PASCollisions_DKESTrajectories`, RHSMode=1, **744,610 unknowns**, one machine, against the
PETSc 3.23 / MUMPS 5.8.2 build of SFINCS v3. Warm is the second solve in a process, after XLA has compiled.

| Configuration | Warm solve | Peak RSS |
| --- | ---: | ---: |
| DKX, `Nxi`-for-`x` ramp | **27.2 s** | **0.93 GB** |
| DKX, uniform `Nxi` | 44.3 s | 1.16 GB |
| DKX, RTX A4000 GPU | 45.0 s | — |
| SFINCS Fortran v3, 1 rank | 463.6 s | 3.98 GB |
| SFINCS Fortran v3, 2 ranks (its best) | 229.5 s | 2.86 GB |

| Cold versus warm, M3 Max CPU | Unknowns | Cold | Warm |
| --- | ---: | ---: | ---: |
| HSX PAS reduced | 40,584 | 1.72 s | 0.12 s |
| HSX PAS, `25x51x100x5` | 744,610 | 23.6 s | 20.0 s |

That is **one measured 744k-unknown HSX PAS case**, chosen because DKX has an exact structured
solver for it. Across all 38 upstream decks: structured route faster on 9 of 9; Krylov route faster
on 7 of 23, six not completed. Every deck, hardware string and method: [performance](docs/performance.rst).

![Measured parity envelopes of DKX against SFINCS Fortran v3](docs/_static/figures/readme/canonical_parity.png)

## Documentation

[Tutorial: first W7-X result](docs/usage.rst) ·
[How-to: resolution, wout files, SFINCS decks](docs/case_files.rst) ·
[Reference: schema, CLI, API](docs/api.rst) ·
[Explanation: physics models, solver routes, limitations](docs/physics_models.rst)

## Cite

```bibtex
@software{dkx,
  author = {Jorge, Rogerio and contributors},
  title  = {DKX: differentiable drift-kinetic neoclassical transport in JAX},
  url    = {https://github.com/uwplasma/DKX},
  year   = {2026}
}
```

Please also cite SFINCS (Landreman, Smith, Mollén & Helander, Phys. Plasmas 21, 042503,
2014) when you use its decks or model. Metadata: [CITATION.cff](CITATION.cff).

[Examples](examples/) · [Research plan](plan.md) · [Contributing](docs/contributing.rst) ·
[SOLVAX](https://github.com/uwplasma/SOLVAX) owns the reusable solver algorithms · [LICENSE](LICENSE)
