"""Exact sampled cubical Betti tensors for three-parameter voxel filtrations.

A filtration value is assigned to each top-dimensional voxel cube.  Every
lower-dimensional cubical cell enters at the coordinatewise minimum of its
incident top-cube values.  For each sampled threshold triple, the implementation
computes B0 and B2 exactly and obtains B1 from the Euler--Poincare identity.
All homology is over F_2.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from .events import compact_dual_births, compact_primal_births, add_birth_histogram
from .graph import bucketize_births, sweep_h0, sweep_h2
from .grid import GridSpec, bin_multifiltration
from .roi import PreparedCubicalROI, prepare_cubical_roi, validate_prepared_roi


def _prepare_bins(
    bins: np.ndarray,
    valid_mask: np.ndarray,
    output_shape: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    b = np.asarray(bins)
    mask = np.asarray(valid_mask, dtype=bool)
    if b.ndim != 4 or b.shape[-1] != 3:
        raise ValueError(f"bins must have shape (X,Y,Z,3), got {b.shape}")
    if mask.shape != b.shape[:3]:
        raise ValueError("valid_mask shape does not match bins")

    shape = tuple(int(v) for v in output_shape)
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError("output_shape must contain three positive integers")
    if np.any(b[mask] < 0):
        raise ValueError("valid voxels cannot have negative bin indices")
    for axis in range(3):
        if np.any(b[..., axis][mask] >= shape[axis]):
            raise ValueError(f"bin index exceeds output axis {axis}")
    return b, mask, shape


def _choose_sweep_axis(output_shape: tuple[int, int, int]) -> int:
    return int(np.argmax(np.asarray(output_shape)))


def _permute_birth_columns(arr: np.ndarray, order: list[int]) -> np.ndarray:
    if order == [0, 1, 2] and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr[:, order], dtype=arr.dtype)


def betti_tensors_from_bins(
    bins: np.ndarray,
    valid_mask: np.ndarray,
    output_shape: Sequence[int],
    *,
    prepared_roi: PreparedCubicalROI | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact ``(B0, B1, B2, chi)`` tensors from discretized top-cube births."""
    b, mask, shape = _prepare_bins(bins, valid_mask, output_shape)
    if prepared_roi is None:
        prepared_roi = prepare_cubical_roi(mask)
    else:
        validate_prepared_roi(prepared_roi, mask)

    mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
    sweep_axis = _choose_sweep_axis(shape)
    order = [sweep_axis] + [q for q in range(3) if q != sweep_axis]
    permuted_shape = tuple(shape[q] for q in order)
    inverse = tuple(order.index(q) for q in range(3))

    delta = np.zeros(shape, dtype=np.int64)

    # Primal graph: B0 from vertices and edges.
    vertex_births, edge_births = compact_primal_births(
        np.ascontiguousarray(b),
        mask_u8,
        prepared_roi.n_vertices,
        prepared_roi.edge_u.shape[0],
    )
    add_birth_histogram(delta, vertex_births, +1)
    add_birth_histogram(delta, edge_births, -1)

    vb = _permute_birth_columns(vertex_births, order)
    eb = _permute_birth_columns(edge_births, order)
    vertex_order, vertex_offsets = bucketize_births(vb, permuted_shape)
    edge_order, edge_offsets = bucketize_births(eb, permuted_shape)
    b0_tmp = sweep_h0(
        vertex_order,
        vertex_offsets,
        edge_order,
        edge_offsets,
        prepared_roi.edge_u,
        prepared_roi.edge_v,
        prepared_roi.n_vertices,
        *permuted_shape,
    )
    b0 = np.transpose(b0_tmp, inverse)

    # Dual complement graph: B2 from cubes and faces.
    cube_births, face_births = compact_dual_births(
        np.ascontiguousarray(b),
        mask_u8,
        prepared_roi.n_valid_cubes,
        prepared_roi.face_u.shape[0],
    )
    add_birth_histogram(delta, face_births, +1)
    add_birth_histogram(delta, cube_births, -1)

    cb = _permute_birth_columns(cube_births, order)
    fb = _permute_birth_columns(face_births, order)
    cube_order, cube_offsets = bucketize_births(cb, permuted_shape)
    face_order, face_offsets = bucketize_births(fb, permuted_shape)
    b2_tmp = sweep_h2(
        cube_order,
        cube_offsets,
        face_order,
        face_offsets,
        prepared_roi.face_u,
        prepared_roi.face_v,
        prepared_roi.n_valid_cubes,
        prepared_roi.n_static_components,
        prepared_roi.n_nodes,
        *permuted_shape,
    )
    b2 = np.transpose(b2_tmp, inverse)

    chi = delta
    for axis in range(3):
        chi = np.cumsum(chi, axis=axis, dtype=np.int64)

    b1 = b0 + b2 - chi
    if np.any(b1 < 0):
        idx = tuple(int(x) for x in np.argwhere(b1 < 0)[0])
        raise RuntimeError(
            f"derived B1 became negative at {idx}: "
            f"B0={b0[idx]}, B2={b2[idx]}, chi={chi[idx]}"
        )
    if not np.array_equal(chi, b0 - b1 + b2):
        raise RuntimeError("Euler-Poincare identity failed")
    return b0, b1, b2, chi


def directional_betti_tensors(
    filtration: np.ndarray,
    valid_mask: np.ndarray | None = None,
    steps: Sequence[int] = (8, 8, 8),
    *,
    return_grid_spec: bool = False,
    prepared_roi: PreparedCubicalROI | None = None,
):
    """Discretize a three-parameter filtration and compute exact Betti tensors."""
    bins, mask, spec = bin_multifiltration(filtration, valid_mask, steps)
    result = betti_tensors_from_bins(
        bins,
        mask,
        spec.output_shape,
        prepared_roi=prepared_roi,
    )
    if return_grid_spec:
        return (*result, spec)
    return result
