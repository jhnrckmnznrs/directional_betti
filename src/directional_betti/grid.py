"""Regular discretization of three-parameter voxel filtrations.

Each filtration coordinate is sampled on ``steps + 1`` threshold values.
Grid bounds are determined by the minimum and maximum filtration values on
valid top-dimensional cubes.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from numba import njit


@dataclass(frozen=True)
class GridSpec:
    steps: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    minima: tuple[float, float, float]
    maxima: tuple[float, float, float]
    step_sizes: tuple[float, float, float]
    dtype: str


def _validate(image: np.ndarray, valid_mask: np.ndarray | None, steps: Sequence[int]):
    arr = np.asarray(image)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected filtration shape (X,Y,Z,3), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError("filtration must use a floating dtype (normally float32)")
    if valid_mask is None:
        mask = np.ones(arr.shape[:3], dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != arr.shape[:3]:
            raise ValueError(f"mask shape {mask.shape} != filtration shape {arr.shape[:3]}")
    if not np.any(mask):
        raise ValueError("valid_mask is empty")
    if len(steps) != 3:
        raise ValueError("this package currently supports exactly 3 filtration parameters")
    st = tuple(int(v) for v in steps)
    if any(v <= 0 for v in st) or any(int(v) != v for v in steps):
        raise ValueError("steps must contain three positive integers")
    return arr, mask, st


@njit(cache=True)
def _scan_minmax_finite(arr: np.ndarray, mask: np.ndarray):
    """Find coordinate minima/maxima over valid top cubes without masked copies."""
    X, Y, Z, _ = arr.shape
    mins = np.empty(3, dtype=arr.dtype)
    maxs = np.empty(3, dtype=arr.dtype)
    found = False
    finite = True
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not mask[x, y, z]:
                    continue
                if not found:
                    for q in range(3):
                        v = arr[x, y, z, q]
                        if not np.isfinite(v):
                            finite = False
                        mins[q] = v
                        maxs[q] = v
                    found = True
                else:
                    for q in range(3):
                        v = arr[x, y, z, q]
                        if not np.isfinite(v):
                            finite = False
                        if v < mins[q]:
                            mins[q] = v
                        if v > maxs[q]:
                            maxs[q] = v
    return mins, maxs, found, finite


@njit(cache=True)
def _fill_bins_streaming(
    arr: np.ndarray,
    mask: np.ndarray,
    mins: np.ndarray,
    step_sizes: np.ndarray,
    steps: np.ndarray,
    out: np.ndarray,
):
    """Assign bins in one streaming pass; return the first invalid mapping."""
    X, Y, Z, _ = arr.shape
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if not mask[x, y, z]:
                    continue
                for q in range(3):
                    # arr/mins retain the filtration dtype for subtraction;
                    # Step sizes are stored in Float64 while input extrema retain their source dtype.
                    numerator = arr[x, y, z, q] - mins[q]
                    idx = int(np.trunc(numerator / step_sizes[q]))
                    if idx < 0 or idx > steps[q]:
                        return x, y, z, q, idx
                    out[x, y, z, q] = idx
    return -1, -1, -1, -1, -1


def _smallest_bin_dtype(steps: tuple[int, int, int]):
    # -1 is reserved for invalid top cubes.
    m = max(steps)
    if m <= np.iinfo(np.int8).max:
        return np.int8
    if m <= np.iinfo(np.int16).max:
        return np.int16
    return np.int32


def bin_multifiltration(
    image: np.ndarray,
    valid_mask: np.ndarray | None = None,
    steps: Sequence[int] = (8, 8, 8),
) -> tuple[np.ndarray, np.ndarray, GridSpec]:
    """Map each valid voxel's three filtration coordinates to regular-grid bins.

    The returned ``bins`` has shape ``(X,Y,Z,3)`` and integer values from
    ``0`` through ``steps[q]``.  Invalid voxels have bin ``-1`` on all axes.

    Grid bounds equal the min/max of valid top-dimensional cubes.
    computes bounds and bins by streaming over the volume instead of creating
    masked/full-volume temporaries.  For the default 8 steps, bins use int8.
    """
    arr, mask, st = _validate(image, valid_mask, steps)
    dtype = np.dtype(arr.dtype)
    scalar = dtype.type

    mins_t, maxs_t, found, finite = _scan_minmax_finite(arr, mask)
    if not found:
        raise ValueError("valid_mask is empty")
    if not finite:
        raise ValueError("valid ROI contains NaN or infinite filtration values")

    step_sizes: list[float] = []
    for q in range(3):
        width = scalar(maxs_t[q]) - scalar(mins_t[q])
        effective = 1.0 if width == 0.0 else width
        step_sizes.append(float(effective / st[q]))

    bins = np.full(arr.shape, -1, dtype=_smallest_bin_dtype(st))
    step_arr = np.asarray(step_sizes, dtype=np.float64)
    st_arr = np.asarray(st, dtype=np.int64)
    bad = _fill_bins_streaming(arr, mask, mins_t, step_arr, st_arr, bins)
    if bad[0] >= 0:
        x, y, z, q, idx = map(int, bad)
        raise IndexError(
            f"filtration value mapped outside grid on axis {q} at {(x, y, z)}: "
            f"index={idx}, allowed=0..{st[q]}"
        )

    spec = GridSpec(
        steps=st,
        output_shape=tuple(v + 1 for v in st),
        minima=tuple(float(x) for x in mins_t),
        maxima=tuple(float(x) for x in maxs_t),
        step_sizes=tuple(step_sizes),
        dtype=str(dtype),
    )
    return bins, mask, spec
