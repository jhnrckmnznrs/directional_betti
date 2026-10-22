"""Public API for exact directional cubical Betti tensors."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .filtration import (
    compute_frame_invariant_covariance,
    foreground_linear_ids,
    frame_invariant_bins_from_covariance_values,
)
from .geometry import prepare_binary_roi
from .grid import GridSpec
from .roi import prepare_cubical_roi
from .topology import betti_tensors_from_bins, directional_betti_tensors


@dataclass(frozen=True)
class BettiTensors:
    """Sampled ``B0``, ``B1``, ``B2`` and Euler-characteristic tensors."""

    b0: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    euler: np.ndarray
    grid: GridSpec

    def assert_consistent(self) -> None:
        """Raise if nonnegativity or the Euler--Poincare identity fails."""
        if np.any(self.b0 < 0) or np.any(self.b1 < 0) or np.any(self.b2 < 0):
            raise AssertionError("Betti tensors must be non-negative")
        if not np.array_equal(self.euler, self.b0 - self.b1 + self.b2):
            raise AssertionError("Euler-Poincare identity failed")


@dataclass(frozen=True)
class DirectionalAnalysis:
    """Frame-invariant results indexed by direction name."""

    by_direction: Mapping[str, BettiTensors]
    spacing: float
    covariance_radius: float
    cone_radius: float
    cone_half_extent: float
    covariance_kernel: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_direction", MappingProxyType(dict(self.by_direction)))


def cubical_betti_tensors(
    filtration: np.ndarray,
    roi: np.ndarray | None = None,
    *,
    grid_steps: Sequence[int] = (8, 8, 8),
) -> BettiTensors:
    """Compute exact sampled Betti tensors for a three-parameter filtration.

    Parameters
    ----------
    filtration:
        Array of shape ``(X, Y, Z, 3)`` assigning one filtration vector to each
        top-dimensional voxel cube.
    roi:
        Optional array of shape ``(X, Y, Z)``. Nonzero cubes belong to the
        cubical domain. If omitted, every top cube belongs to the domain.
    grid_steps:
        Number of equal intervals on each filtration axis. ``(8, 8, 8)``
        produces a ``9 x 9 x 9`` cumulative bin lattice. For a nonconstant
        coordinate, bin ``a < steps`` includes values strictly below its
        upper edge; the terminal bin includes the maximum. Bounds are
        estimated separately for each input.
    """
    b0, b1, b2, chi, spec = directional_betti_tensors(
        filtration,
        roi,
        grid_steps,
        return_grid_spec=True,
    )
    result = BettiTensors(b0=b0, b1=b1, b2=b2, euler=chi, grid=spec)
    result.assert_consistent()
    return result


def frame_invariant_directional_betti(
    foreground: np.ndarray,
    roi: np.ndarray | None = None,
    *,
    spacing: float = 1.0,
    directions: Sequence[str] = ("x", "y", "z"),
    covariance_radius: float,
    cone_radius: float,
    cone_half_extent: float,
    covariance_kernel: str = "gaussian",
    covariance_sigma: float | None = None,
    truncate_sigma: float = 4.0,
    grid_steps: Sequence[int] = (8, 8, 8),
) -> DirectionalAnalysis:
    """Compute frame-invariant directional Betti tensors from a binary volume.

    The three local filtration coordinates are ``(C_d, A_d, S)``: directional
    cone occupancy deficit, directional covariance deficit, and principal-axis
    degeneracy.  ``spacing`` and all radius/extent parameters must use the same
    physical unit.  The current implementation assumes isotropic voxel spacing.
    """
    fg0 = np.asarray(foreground)
    if fg0.ndim != 3:
        raise ValueError(f"foreground must be 3-D, got shape {fg0.shape}")

    if roi is None:
        roi0 = np.ones(fg0.shape, dtype=np.uint8)
    else:
        roi0 = np.asarray(roi)
        if roi0.shape != fg0.shape:
            raise ValueError(f"roi shape {roi0.shape} != foreground shape {fg0.shape}")

    fg, roi_u8 = prepare_binary_roi(fg0, roi0)
    if not np.any(roi_u8):
        raise ValueError("roi is empty")

    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    if not np.isfinite(covariance_radius) or float(covariance_radius) <= 0.0:
        raise ValueError("covariance_radius must be finite and positive")
    if not np.isfinite(cone_radius) or float(cone_radius) < 0.0:
        raise ValueError("cone_radius must be finite and non-negative")
    if not np.isfinite(cone_half_extent) or float(cone_half_extent) <= 0.0:
        raise ValueError("cone_half_extent must be finite and positive")

    dirs = tuple(str(d).lower() for d in directions)
    if not dirs or any(d not in {"x", "y", "z"} for d in dirs):
        raise ValueError("directions must be a non-empty sequence drawn from x, y, z")
    if len(set(dirs)) != len(dirs):
        raise ValueError("directions must not contain duplicates")

    steps = tuple(int(v) for v in grid_steps)
    if (len(steps) != 3 or any(v <= 0 for v in steps)
            or any(int(v) != v for v in grid_steps)):
        raise ValueError("grid_steps must contain three positive integers")

    foreground_ids = foreground_linear_ids(fg, roi_u8)
    covariance_values = compute_frame_invariant_covariance(
        fg,
        roi_u8,
        foreground_ids,
        spacing,
        float(covariance_radius),
        mode=str(covariance_kernel),
        sigma=None if covariance_sigma is None else float(covariance_sigma),
        truncate_sigma=float(truncate_sigma),
    )
    prepared_roi = prepare_cubical_roi(roi_u8)

    results: dict[str, BettiTensors] = {}
    for direction in dirs:
        bins, mask, spec = frame_invariant_bins_from_covariance_values(
            fg,
            roi_u8,
            foreground_ids,
            covariance_values,
            spacing,
            float(cone_radius),
            float(cone_half_extent),
            direction,
            steps,
        )
        b0, b1, b2, chi = betti_tensors_from_bins(
            bins,
            mask,
            spec.output_shape,
            prepared_roi=prepared_roi,
        )
        result = BettiTensors(b0=b0, b1=b1, b2=b2, euler=chi, grid=spec)
        result.assert_consistent()
        results[direction] = result

    return DirectionalAnalysis(
        by_direction=results,
        spacing=spacing,
        covariance_radius=float(covariance_radius),
        cone_radius=float(cone_radius),
        cone_half_extent=float(cone_half_extent),
        covariance_kernel=str(covariance_kernel).lower(),
    )
