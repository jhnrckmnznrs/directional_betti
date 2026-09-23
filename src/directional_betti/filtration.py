"""Frame-invariant filtration construction for binary 3-D volumes.

Local covariance coordinates are evaluated only at foreground voxels.  The
three filtration coordinates are then discretized directly into the sampled
threshold lattice required by the exact cubical topology engine.
"""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from numba import njit, prange

from .grid import GridSpec, _smallest_bin_dtype
from .geometry import (
    DIRECTION_CODES,
    cone_offsets as cached_cone_offsets,
    gaussian_offsets,
    matched_gaussian_sigma,
    prepare_binary_roi,
    sphere_offsets,
)


@njit(cache=True, inline="always")
def _xyz_from_flat(idx: int, Y: int, Z: int):
    yz = Y * Z
    x = idx // yz
    rem = idx - x * yz
    y = rem // Z
    z = rem - y * Z
    return x, y, z


@njit(cache=True, inline="always")
def _in_bounds(x: int, y: int, z: int, X: int, Y: int, Z: int) -> bool:
    return 0 <= x < X and 0 <= y < Y and 0 <= z < Z


@njit(parallel=True, cache=True)
def _frame_invariants_hard(
    grid: np.ndarray,
    valid: np.ndarray,
    foreground_ids: np.ndarray,
    offsets: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Compute covariance-derived coordinates with a hard spherical neighbourhood."""
    X, Y, Z = grid.shape
    out = np.empty((foreground_ids.size, 4), dtype=np.float32)
    eps = 1e-15
    for p in prange(foreground_ids.size):
        x, y, z = _xyz_from_flat(int(foreground_ids[p]), Y, Z)
        n = 0.0
        sx = sy = sz = 0.0
        sxx = sxy = sxz = 0.0
        syy = syz = 0.0
        szz = 0.0
        for i in range(offsets.shape[0]):
            dx = int(offsets[i, 0]); dy = int(offsets[i, 1]); dz = int(offsets[i, 2])
            xx = x + dx; yy = y + dy; zz = z + dz
            if not _in_bounds(xx, yy, zz, X, Y, Z):
                continue
            if valid[xx, yy, zz] == 0 or grid[xx, yy, zz] == 0:
                continue
            n += 1.0
            fx = float(dx) * spacing
            fy = float(dy) * spacing
            fz = float(dz) * spacing
            sx += fx; sy += fy; sz += fz
            sxx += fx*fx; sxy += fx*fy; sxz += fx*fz
            syy += fy*fy; syz += fy*fz; szz += fz*fz

        if n < 2.0:
            out[p, 0] = 1.0; out[p, 1] = 1.0; out[p, 2] = 1.0; out[p, 3] = 1.0
            continue

        mx = sx/n; my = sy/n; mz = sz/n
        C = np.empty((3, 3), dtype=np.float64)
        C[0,0] = sxx/n - mx*mx
        C[0,1] = sxy/n - mx*my
        C[0,2] = sxz/n - mx*mz
        C[1,0] = C[0,1]
        C[1,1] = syy/n - my*my
        C[1,2] = syz/n - my*mz
        C[2,0] = C[0,2]
        C[2,1] = C[1,2]
        C[2,2] = szz/n - mz*mz

        tr = C[0,0] + C[1,1] + C[2,2]
        if tr <= eps:
            out[p, 0] = 1.0; out[p, 1] = 1.0; out[p, 2] = 1.0; out[p, 3] = 1.0
            continue

        qx = min(1.0, max(0.0, C[0,0] / tr))
        qy = min(1.0, max(0.0, C[1,1] / tr))
        qz = min(1.0, max(0.0, C[2,2] / tr))
        out[p, 0] = 1.0 - np.sqrt(qx)
        out[p, 1] = 1.0 - np.sqrt(qy)
        out[p, 2] = 1.0 - np.sqrt(qz)
        vals = np.linalg.eigvalsh(C)
        l1 = max(0.0, vals[2]); l2 = max(0.0, vals[1])
        if l1 <= eps:
            shape = 1.0
        else:
            shape = min(1.0, max(0.0, l2 / l1))
        out[p, 3] = shape
    return out


@njit(parallel=True, cache=True)
def _frame_invariants_weighted(
    grid: np.ndarray,
    valid: np.ndarray,
    foreground_ids: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Compute covariance-derived coordinates with Gaussian neighbourhood weights."""
    X, Y, Z = grid.shape
    out = np.empty((foreground_ids.size, 4), dtype=np.float32)
    eps = 1e-15
    for p in prange(foreground_ids.size):
        x, y, z = _xyz_from_flat(int(foreground_ids[p]), Y, Z)
        sw = 0.0
        n_points = 0
        sx = sy = sz = 0.0
        sxx = sxy = sxz = 0.0
        syy = syz = 0.0
        szz = 0.0
        for i in range(offsets.shape[0]):
            dx = int(offsets[i, 0]); dy = int(offsets[i, 1]); dz = int(offsets[i, 2])
            xx = x + dx; yy = y + dy; zz = z + dz
            if not _in_bounds(xx, yy, zz, X, Y, Z):
                continue
            if valid[xx, yy, zz] == 0 or grid[xx, yy, zz] == 0:
                continue
            w = float(weights[i])
            if w <= 0.0:
                continue
            fx = float(dx) * spacing
            fy = float(dy) * spacing
            fz = float(dz) * spacing
            sw += w; n_points += 1
            sx += w*fx; sy += w*fy; sz += w*fz
            sxx += w*fx*fx; sxy += w*fx*fy; sxz += w*fx*fz
            syy += w*fy*fy; syz += w*fy*fz; szz += w*fz*fz

        if n_points < 2 or sw <= eps:
            out[p, 0] = 1.0; out[p, 1] = 1.0; out[p, 2] = 1.0; out[p, 3] = 1.0
            continue

        mx = sx/sw; my = sy/sw; mz = sz/sw
        C = np.empty((3, 3), dtype=np.float64)
        C[0,0] = sxx/sw - mx*mx
        C[0,1] = sxy/sw - mx*my
        C[0,2] = sxz/sw - mx*mz
        C[1,0] = C[0,1]
        C[1,1] = syy/sw - my*my
        C[1,2] = syz/sw - my*mz
        C[2,0] = C[0,2]
        C[2,1] = C[1,2]
        C[2,2] = szz/sw - mz*mz

        tr = C[0,0] + C[1,1] + C[2,2]
        if tr <= eps:
            out[p, 0] = 1.0; out[p, 1] = 1.0; out[p, 2] = 1.0; out[p, 3] = 1.0
            continue

        qx = min(1.0, max(0.0, C[0,0] / tr))
        qy = min(1.0, max(0.0, C[1,1] / tr))
        qz = min(1.0, max(0.0, C[2,2] / tr))
        out[p, 0] = 1.0 - np.sqrt(qx)
        out[p, 1] = 1.0 - np.sqrt(qy)
        out[p, 2] = 1.0 - np.sqrt(qz)
        vals = np.linalg.eigvalsh(C)
        l1 = max(0.0, vals[2]); l2 = max(0.0, vals[1])
        if l1 <= eps:
            shape = 1.0
        else:
            shape = min(1.0, max(0.0, l2 / l1))
        out[p, 3] = shape
    return out


@njit(parallel=True, cache=True)
def _cone_values(
    grid: np.ndarray,
    valid: np.ndarray,
    foreground_ids: np.ndarray,
    cone_offsets_z: np.ndarray,
    cone_weights: np.ndarray,
    direction_code: int,
) -> np.ndarray:
    """Cone coordinate C_d only at foreground voxels."""
    X, Y, Z = grid.shape
    out = np.empty(foreground_ids.size, dtype=np.float32)
    for p in prange(foreground_ids.size):
        x, y, z = _xyz_from_flat(int(foreground_ids[p]), Y, Z)
        full = 0.0
        valid_weight = 0.0
        for i in range(cone_offsets_z.shape[0]):
            a = int(cone_offsets_z[i, 0]); b = int(cone_offsets_z[i, 1]); c = int(cone_offsets_z[i, 2])
            if direction_code == 0:
                dx = c; dy = a; dz = b
            elif direction_code == 1:
                dx = a; dy = c; dz = b
            else:
                dx = a; dy = b; dz = c
            xx = x + dx; yy = y + dy; zz = z + dz
            if not _in_bounds(xx, yy, zz, X, Y, Z):
                continue
            if valid[xx, yy, zz] == 0:
                continue
            w = cone_weights[i]
            valid_weight += w
            if grid[xx, yy, zz] != 0:
                full += w
        out[p] = 1.0 if valid_weight == 0.0 else 1.0 - full/valid_weight
    return out


@njit(cache=True)
def _filtration_minmax(cone: np.ndarray, pca: np.ndarray, pca_axis: int, include_sentinel: bool):
    sentinel = np.float32(1.25)
    mins = np.empty(3, dtype=np.float32)
    maxs = np.empty(3, dtype=np.float32)
    found = False
    finite = True
    if include_sentinel:
        mins[0] = sentinel; mins[1] = sentinel; mins[2] = sentinel
        maxs[0] = sentinel; maxs[1] = sentinel; maxs[2] = sentinel
        found = True
    for i in range(cone.size):
        v0 = cone[i]; v1 = pca[i, pca_axis]; v2 = pca[i, 3]
        if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2)):
            finite = False
        if not found:
            mins[0] = v0; maxs[0] = v0
            mins[1] = v1; maxs[1] = v1
            mins[2] = v2; maxs[2] = v2
            found = True
        else:
            if v0 < mins[0]: mins[0] = v0
            if v0 > maxs[0]: maxs[0] = v0
            if v1 < mins[1]: mins[1] = v1
            if v1 > maxs[1]: maxs[1] = v1
            if v2 < mins[2]: mins[2] = v2
            if v2 > maxs[2]: maxs[2] = v2
    return mins, maxs, found, finite


@njit(cache=True)
def _fill_bins(
    valid: np.ndarray,
    foreground_ids: np.ndarray,
    cone: np.ndarray,
    pca: np.ndarray,
    pca_axis: int,
    mins: np.ndarray,
    step_sizes: np.ndarray,
    steps: np.ndarray,
    include_sentinel: bool,
    out: np.ndarray,
):
    X, Y, Z = valid.shape
    sentinel = np.float32(1.25)
    # Fill marrow top cubes with the sentinel bin only when marrow exists.
    # If the ROI is entirely foreground, the sentinel lies outside the observed grid and
    # must not participate in either bounds or bin assignment.
    if include_sentinel:
        sb0 = int(np.trunc((sentinel - mins[0]) / step_sizes[0]))
        sb1 = int(np.trunc((sentinel - mins[1]) / step_sizes[1]))
        sb2 = int(np.trunc((sentinel - mins[2]) / step_sizes[2]))
        if sb0 < 0 or sb0 > steps[0] or sb1 < 0 or sb1 > steps[1] or sb2 < 0 or sb2 > steps[2]:
            return -2, -2, -2, -2, -2
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    if valid[x, y, z] != 0:
                        out[x, y, z, 0] = sb0
                        out[x, y, z, 1] = sb1
                        out[x, y, z, 2] = sb2
    for i in range(foreground_ids.size):
        idx = int(foreground_ids[i])
        x, y, z = _xyz_from_flat(idx, Y, Z)
        vals0 = cone[i]
        vals1 = pca[i, pca_axis]
        vals2 = pca[i, 3]
        b0 = int(np.trunc((vals0 - mins[0]) / step_sizes[0]))
        b1 = int(np.trunc((vals1 - mins[1]) / step_sizes[1]))
        b2 = int(np.trunc((vals2 - mins[2]) / step_sizes[2]))
        if b0 < 0 or b0 > steps[0]: return x, y, z, 0, b0
        if b1 < 0 or b1 > steps[1]: return x, y, z, 1, b1
        if b2 < 0 or b2 > steps[2]: return x, y, z, 2, b2
        out[x, y, z, 0] = b0; out[x, y, z, 1] = b1; out[x, y, z, 2] = b2
    return -1, -1, -1, -1, -1


def foreground_linear_ids(grid: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return compact linear IDs of foreground voxels inside the ROI."""
    g, m = prepare_binary_roi(grid, valid_mask)
    ids64 = np.flatnonzero((g.ravel() != 0) & (m.ravel() != 0))
    if ids64.size and int(ids64[-1]) > np.iinfo(np.int32).max:
        return ids64.astype(np.int64, copy=False)
    return ids64.astype(np.int32, copy=False)


def compute_frame_invariant_covariance(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    foreground_ids: np.ndarray,
    spacing: float,
    radius: float,
    *,
    mode: str = "hard",
    sigma: float | None = None,
    truncate_sigma: float = 4.0,
) -> np.ndarray:
    """Compute ``(A_x,A_y,A_z,S)`` only at foreground voxels."""
    g, m = prepare_binary_roi(grid, valid_mask)
    ids = np.asarray(foreground_ids)
    mode = str(mode).lower()
    if mode == "hard":
        offsets = sphere_offsets(radius, spacing)
        return _frame_invariants_hard(g, m, ids, offsets, float(spacing))
    if mode == "gaussian":
        sigma = matched_gaussian_sigma(radius) if sigma is None else float(sigma)
        offsets, weights = gaussian_offsets(
            sigma, float(spacing), float(truncate_sigma)
        )
        return _frame_invariants_weighted(
            g, m, ids, offsets, weights, float(spacing)
        )
    raise ValueError("covariance mode must be one of: hard, gaussian")


def frame_invariant_bins_from_covariance_values(
    grid: np.ndarray,
    valid_mask: np.ndarray,
    foreground_ids: np.ndarray,
    covariance_values: np.ndarray,
    spacing: float,
    cone_radius: float,
    cone_half_extent: float,
    direction: str,
    steps: Sequence[int] = (8, 8, 8),
) -> tuple[np.ndarray, np.ndarray, GridSpec]:
    """Build exact frame-invariant bins without a full float filtration volume."""
    g, m = prepare_binary_roi(grid, valid_mask)
    ids = np.asarray(foreground_ids)
    p = np.asarray(covariance_values, dtype=np.float32)
    if p.shape != (ids.size, 4):
        raise ValueError(f"covariance_values shape {p.shape} incompatible with {ids.size} foreground ids")
    d = str(direction).lower()
    if d not in DIRECTION_CODES:
        raise ValueError("direction must be one of: x, y, z")
    if len(steps) != 3:
        raise ValueError("steps must contain three positive integers")
    st = tuple(int(v) for v in steps)
    if any(v <= 0 for v in st) or any(int(v) != v for v in steps):
        raise ValueError("steps must contain three positive integers")

    cone_kernel_offsets, cone_weights = cached_cone_offsets(
        cone_radius, cone_half_extent, spacing
    )
    cone = _cone_values(
        g, m, ids, cone_kernel_offsets, cone_weights, DIRECTION_CODES[d]
    )
    pca_axis = DIRECTION_CODES[d]
    n_roi = int(np.count_nonzero(m))
    include_sentinel = n_roi > ids.size
    mins, maxs, found, finite = _filtration_minmax(cone, p, pca_axis, include_sentinel)
    if not found:
        raise ValueError("valid_mask is empty")
    if not finite:
        raise ValueError("valid ROI contains NaN or infinite filtration values")

    step_sizes: list[float] = []
    scalar = np.float32
    for q in range(3):
        width = scalar(maxs[q]) - scalar(mins[q])
        effective = 1.0 if width == 0.0 else width
        step_sizes.append(float(effective / st[q]))
    step_arr = np.asarray(step_sizes, dtype=np.float64)
    st_arr = np.asarray(st, dtype=np.int64)
    bins = np.full(g.shape + (3,), -1, dtype=_smallest_bin_dtype(st))
    bad = _fill_bins(
        m, ids, cone, p, pca_axis, mins, step_arr, st_arr, include_sentinel, bins
    )
    if bad[0] >= 0 or bad[0] == -2:
        x, y, z, q, idx = map(int, bad)
        raise IndexError(
            f"filtration value mapped outside grid on axis {q} at {(x, y, z)}: "
            f"index={idx}, allowed=0..{st[q] if q >= 0 else '?'}"
        )
    spec = GridSpec(
        steps=st,
        output_shape=tuple(v + 1 for v in st),
        minima=tuple(float(x) for x in mins),
        maxima=tuple(float(x) for x in maxs),
        step_sizes=tuple(step_sizes),
        dtype=str(np.dtype(np.float32)),
    )
    return bins, np.asarray(m, dtype=bool), spec
