# DKX

[![PyPI](https://img.shields.io/pypi/v/dkx)](https://pypi.org/project/dkx/)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/DKX/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/DKX/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/sfincs-jax?label=docs)](https://sfincs-jax.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/uwplasma/DKX)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)

Neoclassical transport for stellarators and tokamaks, in JAX.

DKX solves the radially local, linearized drift-kinetic equation on a flux
surface and returns particle and heat fluxes, parallel flows, bootstrap
current, transport matrices, and ambipolar electric-field roots. Matched cases are
tested against [SFINCS Fortran v3](https://github.com/landreman/sfincs).
Expert JAX paths support implicit derivatives in documented domains.
See [capability scope](docs/capabilities.rst) for native workflow and root-selection
limits, and the [research roadmap](plan.md).

![W7-X standard configuration solved by DKX: the boundary colored by |B| and by parallel current density, the bootstrap current profile, and the ambipolar radial electric field](docs/_static/figures/readme/w7x_showcase.png)

*W7-X standard configuration, every panel a DKX solve output. Boundary colored
by `|B|` and by parallel current density `j‖(θ, ζ)`; bootstrap current profile;
ambipolar `E_r` against Pablant et al., Phys. Plasmas 25, 022508.*

## Install

```bash
pip install dkx                      # CPU
pip install -U "jax[cuda12]"         # add for GPU
```

## Run

```bash
dkx schema --format toml > case.toml   # commented template
dkx run case.toml --out result.nc      # solve
dkx inspect result.nc                  # what the result holds
```

```python
import dkx

case = dkx.Case.from_mapping({
    "schema": 1,
    "name": "tokamak",
    "run": {"workflow": "profile", "progress": False},
    "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.16, 0.25, 0.36]},
    "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                 "density_m3": [8.0e19, 7.0e19, 6.0e19],
                 "temperature_keV": [1.0, 0.8, 0.6]}],
    "physics": {"model": "full_local", "collisions": "pitch_angle_scattering",
                "magnetic_drifts": "dkes", "phi1": "off"},
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
    "solver": {"method": "auto", "relative_tolerance": 1e-8},
})
result = dkx.run(case)

print("solver route:", result.metadata["solver_route"])
print("particle flux:", float(result.arrays["particle_flux_m2_s"][1, 0]))
```

A case is one TOML or JSON file. `Case` is immutable with a deterministic
`case_id`; `Result` carries the arrays, solver route, achieved residual, and
provenance. See [case files](docs/case_files.rst).

The resolution above is sized to run in seconds, and it is not converged —
`dkx converge` reports this case still moving by more than 100% under
refinement. Check any case before trusting its numbers:

```bash
dkx converge case.toml     # refines theta, zeta, pitch and speed
```

It refines the axes jointly as well as one at a time, because one at a time
can mislead: on the shipped tokamak example the theta axis looks settled to
0.2% at `pitch = 8` and moves the outputs 74% at `pitch = 40`.

## Speed

![Runtime and peak memory, DKX against SFINCS Fortran v3, on the 744k-unknown HSX PAS case](docs/_static/figures/readme/tier1_hsx_runtime_memory.png)

`HSX_PASCollisions_DKESTrajectories`, RHSMode=1, **744,610 unknowns**, one
machine, against the PETSc 3.23 and MUMPS 5.8.2 build of SFINCS v3.

| Configuration | Warm solve | Peak RSS |
| --- | ---: | ---: |
| DKX, `Nxi`-for-`x` ramp | **27.2 s** | **0.93 GB** |
| DKX, uniform `Nxi` | 44.3 s | 1.16 GB |
| DKX, RTX A4000 GPU | 45.0 s | — |
| SFINCS Fortran v3, 1 rank | 463.6 s | 3.98 GB |
| SFINCS Fortran v3, 2 ranks | 229.5 s | 2.86 GB |

Two ranks is the Fortran build's best; 4 and 8 are slower.

### Cold and warm solves

The table above is the **warm** solve: the second onward in a process, after
XLA has compiled. An optimizer or `Er` scan sees that; one terminal run does
not. Both, on an Apple M3 Max CPU with `JAX_ENABLE_X64`:

| Case | Unknowns | Cold | Warm | Cold / warm |
| --- | ---: | ---: | ---: | ---: |
| HSX PAS reduced | 40,584 | 1.72 s | 0.12 s | 14x |
| HSX PAS, `25x51x100x5` | 744,610 | 23.6 s | 20.0 s | 1.18x |

Compilation costs the same either way, so it dominates a small run and
disappears into a large one: timing DKX on a toy case measures XLA, not the
solver. The headline does not rest on the warm number: cold, that case is
23.6 s against the Fortran build's 463.6 s and 229.5 s, both cold by construction.

That is **one measured 744k-unknown HSX PAS case**, picked because DKX has a
structured direct solver for it. Across all 38 upstream decks, structure rather
than problem size decides the outcome:

| Across 38 upstream decks | DKX |
| --- | --- |
| Structured direct route faster | 9 of 9 |
| Recycled Krylov route faster | 7 of 23 |
| Lower peak memory | 3 of 32 |
| Did not complete | 6 (reference: 0 of 38) |
| Median agreement | `4.1e-06` |

Losses sit where the block-tridiagonal structure breaks: Fokker-Planck
collisions, magnetic drifts, `Er` `xDot`/`xiDot` terms, `Phi1` iteration.
Full tables: [performance](docs/performance.rst).

## Accuracy

![Measured parity envelopes of DKX against SFINCS Fortran v3](docs/_static/figures/readme/canonical_parity.png)

Every canonical module is pinned against SFINCS Fortran v3 in CI: outputs,
per-species tables, and console prints match field by field.

One exception, documented rather than hidden: the scheme-1 monoenergetic
`transportMatrix[0,1]` element is tolerance-unstable in the Fortran build
itself, so it is pinned to upstream's expected value, which DKX reproduces to
`4.2e-6`.

### Against SFINCS, MONKES and YANCC

![DKX against SFINCS Fortran v3, MONKES and YANCC: scaled differences at 1e-10 to 1e-8 on matched full Fokker-Planck decks, and percent-level agreement on Beidler-normalized monoenergetic coefficients](docs/_static/figures/readme/cross_code_validation.png)

Two comparisons that mean different things, so they are reported separately.

Against **SFINCS Fortran v3** the equations and the discretization are the
same, so the only question is whether the JAX reimplementation reproduces the
Fortran arithmetic. On matched decks with full Fokker-Planck collisions it does,
to `2.7e-10` on an axisymmetric surface, `1.9e-09` with a finite electric field,
and `1.4e-08` on a stellarator — differences of solver tolerance and summation
order, not of physics. The W7-X row is the strict one at `3.2e-03`, because it
runs the whole native path through to physical flux units.

Against **MONKES** and **YANCC** the codes are independent, with their own
discretizations, compared through the Beidler normalization. All four
coefficients agree within the 6% release gate on all three configurations, and
`D33` — the Spitzer-normalized parallel conductivity — agrees to better than
0.1% everywhere.

The cross-code rung is deliberately bounded: matched zero-field monoenergetic
PAS/DKES equations only. It is not a finite-`Er`, ambipolar-profile,
experimental, or performance comparison, and the artifact says so in five
explicit exclusions.

Every number in that figure is read at build time out of the sealed artifacts
under [`validation/`](validation/) — the same files that gate a release, with
pinned upstream commits for
[SFINCS](https://github.com/landreman/sfincs),
[MONKES](https://github.com/JavierEscoto/MONKES) and
[YANCC](https://github.com/f0uriest/yancc). Regenerate with
`python tools/benchmarks/cross_code_readme_figure.py`; it fails rather than
draw a stale value. Full matrix: [validation](docs/validation_matrix.rst).

## Capabilities

| | DKX | SFINCS v3 |
| --- | :---: | :---: |
| RHSMode 1/2/3: fluxes, flows, bootstrap current, transport matrices | ✅ | ✅ |
| Pitch-angle and full Fokker-Planck (Rosenbluth) collisions | ✅ | ✅ |
| Geometry: analytic 1-4, VMEC 5, Boozer `.bc` 11/12, spectrum 13, `lasym` | ✅ | ✅ |
| `Phi1` quasineutrality; Tangential magnetic drifts; `export_f` | ✅ | ✅ |
| Ambipolar radial-electric-field root solve | ✅ | ✅ |
| Exact gradients of any output w.r.t. any input (`jax.grad`) | ✅ | ❌ |
| GPU; warm starts and Krylov recycling across scans | ✅ | ❌ |

## Limitations

- Admitted W7-X ambipolar roots cover explicit sampled intervals only. A finite
  sign-sampled grid cannot exclude an even number of crossings, so "no bracket
  found" never means "no root".
- W7-X bootstrap and parallel current are not converged in `theta`. Only
  particle and heat flux are admitted at the tested grid.
- Case execution does not yet cover declarative scans, transport-matrix
  results, or explicit sharding. Unsupported combinations raise rather than
  substitute another model.
- The 0.93 GB figure above has not reproduced on a second registered host
  (`validation/benchmarks/`).

## SFINCS compatibility

DKX reads SFINCS `input.namelist` decks and writes SFINCS-keyed HDF5 and
NetCDF, permanently.

```bash
dkx input.namelist --out sfincsOutput.h5
dkx --plot sfincsOutput.h5             # panels from a DKX or Fortran run
dkx wout_XXX.nc                        # equilibrium in, panels out
```

## From an equilibrium

`dkx wout_XXX.nc` is a survey, not a study. It runs in about a minute and
produces panels, and it has to invent a plasma to do so: a VMEC equilibrium
fixes the **pressure** and nothing else.

DKX scales the on-axis density from that pressure against the published
reactor profiles of [Landreman, Buller and Drevlak
(2022)](https://arxiv.org/abs/2205.02914), then takes the temperature from
`p = 2nT`. The run prints the pair it used. Pin the density to your own design
point when you have one:

```bash
dkx wout_XXX.nc --density-m3 2.38e20   # T(0) then follows from p = 2nT
```

The bootstrap current is sensitive to this: at fixed pressure it changes by a
factor of five across a plausible density range, because collisionality goes
like `n/T²`. A bootstrap current from a pressure profile alone cannot be
compared against one from an optimizer that assumed a different plasma.

For real work, write a [case file](docs/case_files.rst) with your own profiles
and resolution, then check it:

```bash
dkx validate case.toml                 # will it run, and what is its id
dkx converge case.toml                 # is the resolution enough
dkx run case.toml --out result.nc
```

## Documentation

[Case files](docs/case_files.rst) · [Install](docs/installation.rst) ·
[Usage](docs/usage.rst) · [Examples](docs/examples.rst) ·
[Physics](docs/physics_models.rst) ·
[Performance](docs/performance.rst) ·
[Validation](docs/validation_matrix.rst)

## License

See [LICENSE](LICENSE).
