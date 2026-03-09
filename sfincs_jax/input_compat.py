from __future__ import annotations

from typing import Any, Mapping


def _group_get(group: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = group.get(key.upper(), None)
        if value is not None:
            return value
    return None


def effective_equilibrium_file(*, geom_params: Mapping[str, Any]) -> Any | None:
    geometry_scheme = int(_group_get(geom_params, "geometryScheme") or -1)
    equilibrium_file = _group_get(geom_params, "equilibriumFile")
    if equilibrium_file is not None:
        return equilibrium_file
    if geometry_scheme == 10:
        return _group_get(geom_params, "fort996boozer_file")
    if geometry_scheme == 11:
        return _group_get(geom_params, "JGboozer_file")
    if geometry_scheme == 12:
        return _group_get(geom_params, "JGboozer_file_NonStelSym")
    return None


def effective_r_n_wish(*, geom_params: Mapping[str, Any], default: float = 0.5) -> float:
    value = _group_get(geom_params, "rN_wish", "normradius_wish")
    return float(value) if value is not None else float(default)


def effective_psi_a_hat(
    *,
    geom_params: Mapping[str, Any],
    phys_params: Mapping[str, Any],
    default: float,
) -> float:
    value = _group_get(geom_params, "psiAHat")
    if value is None:
        value = _group_get(phys_params, "psiAHat")
    return float(value) if value is not None else float(default)


def infer_species_input_radial_coordinate_for_gradients(
    *,
    geom_params: Mapping[str, Any],
    species_params: Mapping[str, Any],
    default: int = 4,
) -> int:
    explicit = _group_get(geom_params, "inputRadialCoordinateForGradients")
    if explicit is not None:
        return int(explicit)

    if _group_get(species_params, "dNHatdrHats", "dTHatdrHats") is not None:
        return 2
    if _group_get(species_params, "dNHatdpsiHats", "dTHatdpsiHats") is not None:
        return 0
    if _group_get(species_params, "dNHatdpsiNs", "dTHatdpsiNs") is not None:
        return 1
    if _group_get(species_params, "dNHatdrNs", "dTHatdrNs") is not None:
        return 3
    return int(default)


def infer_phi_input_radial_coordinate_for_gradients(
    *,
    geom_params: Mapping[str, Any],
    phys_params: Mapping[str, Any],
    default: int = 4,
) -> int:
    explicit = _group_get(geom_params, "inputRadialCoordinateForGradients")
    if explicit is not None:
        return int(explicit)

    if _group_get(phys_params, "Er") is not None:
        return 4
    if _group_get(phys_params, "dPhiHatdrHat") is not None:
        return 2
    if _group_get(phys_params, "dPhiHatdpsiHat") is not None:
        return 0
    if _group_get(phys_params, "dPhiHatdpsiN") is not None:
        return 1
    if _group_get(phys_params, "dPhiHatdrN") is not None:
        return 3
    return int(default)


def infer_input_radial_coordinate_for_gradients(
    *,
    geom_params: Mapping[str, Any],
    species_params: Mapping[str, Any],
    phys_params: Mapping[str, Any],
    default: int = 4,
) -> int:
    explicit = _group_get(geom_params, "inputRadialCoordinateForGradients")
    if explicit is not None:
        return int(explicit)

    phi_coord = infer_phi_input_radial_coordinate_for_gradients(
        geom_params=geom_params,
        phys_params=phys_params,
        default=default,
    )
    if _group_get(phys_params, "dPhiHatdpsiHat", "dPhiHatdpsiN", "dPhiHatdrHat", "dPhiHatdrN", "Er") is not None:
        return int(phi_coord)

    return infer_species_input_radial_coordinate_for_gradients(
        geom_params=geom_params,
        species_params=species_params,
        default=default,
    )


def effective_use_iterative_linear_solver(*, other_params: Mapping[str, Any], default: int = 1) -> int:
    value = _group_get(other_params, "useIterativeLinearSolver", "useIterativeSolver")
    return int(value) if value is not None else int(default)
