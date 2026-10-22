"""Geometric neighbourhood construction for frame-invariant filtrations."""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from numba import njit

DIRECTION_CODES = {"x": 0, "y": 1, "z": 2}


@njit(cache=True)
def _has_foreground_outside_roi(foreground: np.ndarray, roi: np.ndarray) -> bool:
    fg = foreground.ravel()
    mask = roi.ravel()
    for i in range(fg.size):
        if fg[i] != 0 and mask[i] == 0:
            return True
    return False


def prepare_binary_roi(
    foreground: np.ndarray,
    roi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching binary ``uint8`` foreground and ROI arrays.

    Nonzero values are treated as foreground/ROI membership.  Foreground values
    outside the ROI are removed.
    """
    fg0 = np.asarray(foreground)
    roi0 = np.asarray(roi)
    if fg0.ndim != 3 or roi0.ndim != 3 or fg0.shape != roi0.shape:
        raise ValueError(
            "foreground and roi must be matching 3-D arrays, "
            f"got {fg0.shape} and {roi0.shape}"
        )

    if fg0.dtype == np.uint8 and (fg0.size == 0 or int(fg0.max()) <= 1):
        fg = fg0
    else:
        fg = (fg0 != 0).astype(np.uint8, copy=False)

    if roi0.dtype == np.uint8 and (roi0.size == 0 or int(roi0.max()) <= 1):
        mask = roi0
    else:
        mask = (roi0 != 0).astype(np.uint8, copy=False)

    if _has_foreground_outside_roi(fg, mask):
        fg = np.where(mask != 0, fg, 0).astype(np.uint8, copy=False)
    return fg, mask


def make_sphere_offsets(radius: float, spacing: float) -> np.ndarray:
    """Integer offsets whose physical Euclidean distance is at most ``radius``."""
    radius = float(radius)
    spacing = float(spacing)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")

    max_off = int(np.floor(radius / spacing))
    radius2 = radius * radius
    offsets: list[tuple[int, int, int]] = []
    for dx in range(-max_off, max_off + 1):
        x2 = (dx * spacing) ** 2
        for dy in range(-max_off, max_off + 1):
            y2 = (dy * spacing) ** 2
            for dz in range(-max_off, max_off + 1):
                if x2 + y2 + (dz * spacing) ** 2 <= radius2 + 1e-12:
                    offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=np.int16)


def make_cone_offsets(
    radius: float,
    half_extent: float,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Offsets of the symmetric double cone used by the occupancy coordinate."""
    radius = float(radius)
    half_extent = float(abs(half_extent))
    spacing = float(spacing)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("cone radius must be finite and non-negative")
    if not np.isfinite(half_extent):
        raise ValueError("cone half extent must be finite")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    if half_extent == 0.0:
        return np.zeros((0, 3), dtype=np.int16), np.zeros(0, dtype=np.float64)

    max_xy = int(np.floor(radius / spacing))
    max_z = int(np.floor(half_extent / spacing))
    offsets: list[tuple[int, int, int]] = []
    weights: list[float] = []
    for dx in range(-max_xy, max_xy + 1):
        for dy in range(-max_xy, max_xy + 1):
            transverse = ((dx * spacing) ** 2 + (dy * spacing) ** 2) ** 0.5
            for dz in range(-max_z, max_z + 1):
                axial = abs(dz) * spacing
                if (
                    axial <= half_extent + 1e-12
                    and transverse <= radius * axial / half_extent + 1e-12
                ):
                    offsets.append((dx, dy, dz))
                    weights.append(1.0)
    return np.asarray(offsets, dtype=np.int16), np.asarray(weights, dtype=np.float64)


def matched_gaussian_sigma(radius: float) -> float:
    """Gaussian sigma with the same radial second moment as a uniform 3-D ball."""
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    return radius / np.sqrt(5.0)


def make_gaussian_offsets(
    sigma: float,
    spacing: float,
    truncate_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-weighted voxel offsets truncated at ``truncate_sigma * sigma``."""
    sigma = float(sigma)
    spacing = float(spacing)
    truncate_sigma = float(truncate_sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    if not np.isfinite(truncate_sigma) or truncate_sigma < 2.0:
        raise ValueError("truncate_sigma must be finite and >= 2")

    cutoff = truncate_sigma * sigma
    max_off = int(np.floor(cutoff / spacing))
    cutoff2 = cutoff * cutoff
    denominator = 2.0 * sigma * sigma
    offsets: list[tuple[int, int, int]] = []
    weights: list[float] = []
    for dx in range(-max_off, max_off + 1):
        x2 = (dx * spacing) ** 2
        for dy in range(-max_off, max_off + 1):
            y2 = (dy * spacing) ** 2
            for dz in range(-max_off, max_off + 1):
                r2 = x2 + y2 + (dz * spacing) ** 2
                if r2 <= cutoff2 + 1e-12:
                    offsets.append((dx, dy, dz))
                    weights.append(float(np.exp(-r2 / denominator)))
    return np.asarray(offsets, dtype=np.int16), np.asarray(weights, dtype=np.float64)


@lru_cache(maxsize=256)
def sphere_offsets(radius: float, spacing: float) -> np.ndarray:
    return make_sphere_offsets(float(radius), float(spacing))


@lru_cache(maxsize=256)
def cone_offsets(radius: float, half_extent: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    return make_cone_offsets(float(radius), float(half_extent), float(spacing))


@lru_cache(maxsize=256)
def gaussian_offsets(
    sigma: float,
    spacing: float,
    truncate_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    return make_gaussian_offsets(float(sigma), float(spacing), float(truncate_sigma))
