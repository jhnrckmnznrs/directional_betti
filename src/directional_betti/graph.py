"""Bucketed union-find sweeps for exact sampled cubical B0 and B2."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _find(uf: np.ndarray, node: int) -> int:
    """Find a union-find root using path halving."""
    while uf[node] >= 0:
        parent = uf[node]
        if uf[parent] >= 0:
            uf[node] = uf[parent]
        node = parent
    return node


@njit(cache=True, inline="always")
def _union(uf: np.ndarray, a: int, b: int) -> int:
    """Union two nodes; return 1 exactly when two components merge."""
    ra = _find(uf, a)
    rb = _find(uf, b)
    if ra == rb:
        return 0
    # Negative root values store component sizes.
    if uf[ra] > uf[rb]:
        ra, rb = rb, ra
    uf[ra] += uf[rb]
    uf[rb] = ra
    return 1


@njit(cache=True)
def _bucketize(births: np.ndarray, n0: int, n1: int, n2: int):
    n = births.shape[0]
    n_buckets = n0 * n1 * n2
    counts = np.zeros(n_buckets, dtype=np.int64)
    for i in range(n):
        key = (int(births[i, 0]) * n1 + int(births[i, 1])) * n2 + int(births[i, 2])
        counts[key] += 1

    offsets = np.empty(n_buckets + 1, dtype=np.int64)
    offsets[0] = 0
    for key in range(n_buckets):
        offsets[key + 1] = offsets[key] + counts[key]

    cursor = offsets[:-1].copy()
    order = np.empty(n, dtype=np.int32)
    for i in range(n):
        key = (int(births[i, 0]) * n1 + int(births[i, 1])) * n2 + int(births[i, 2])
        pos = cursor[key]
        order[pos] = i
        cursor[key] = pos + 1
    return order, offsets


def bucketize_births(births: np.ndarray, shape: tuple[int, int, int]):
    """Group event indices by their three-dimensional integer birth value."""
    b = np.asarray(births)
    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError("births must have shape (N,3)")
    if b.dtype not in (np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32)):
        b = b.astype(np.int16, copy=False)
    b = np.ascontiguousarray(b)

    n0, n1, n2 = map(int, shape)
    if b.size and (
        np.any(b < 0)
        or np.any(b[:, 0] >= n0)
        or np.any(b[:, 1] >= n1)
        or np.any(b[:, 2] >= n2)
    ):
        raise ValueError("birth outside bucket grid")
    return _bucketize(b, n0, n1, n2)


@njit(cache=True, parallel=True)
def sweep_h0(
    vertex_order: np.ndarray,
    vertex_offsets: np.ndarray,
    edge_order: np.ndarray,
    edge_offsets: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    n_vertices: int,
    n0: int,
    n1: int,
    n2: int,
) -> np.ndarray:
    """Count connected components of the present primal 1-skeleton."""
    out = np.zeros((n0, n1, n2), dtype=np.int64)
    for pair in prange(n1 * n2):
        j = pair // n2
        k = pair - j * n2
        uf = np.empty(n_vertices, dtype=np.int32)
        components = 0

        for a in range(n0):
            for q in range(j + 1):
                base = (a * n1 + q) * n2
                for r in range(k + 1):
                    bucket = base + r
                    for pos in range(vertex_offsets[bucket], vertex_offsets[bucket + 1]):
                        node = vertex_order[pos]
                        uf[node] = -1
                        components += 1

            for q in range(j + 1):
                base = (a * n1 + q) * n2
                for r in range(k + 1):
                    bucket = base + r
                    for pos in range(edge_offsets[bucket], edge_offsets[bucket + 1]):
                        edge = edge_order[pos]
                        components -= _union(uf, edge_u[edge], edge_v[edge])
            out[a, j, k] = components
    return out


@njit(cache=True, parallel=True)
def sweep_h2(
    cube_order: np.ndarray,
    cube_offsets: np.ndarray,
    face_order: np.ndarray,
    face_offsets: np.ndarray,
    face_u: np.ndarray,
    face_v: np.ndarray,
    n_valid_cubes: int,
    n_static_components: int,
    n_nodes: int,
    n0: int,
    n1: int,
    n2: int,
) -> np.ndarray:
    """Count bounded complement components in the contracted dual graph."""
    out = np.zeros((n0, n1, n2), dtype=np.int64)
    for pair in prange(n1 * n2):
        j = pair // n2
        k = pair - j * n2
        uf = np.empty(n_nodes, dtype=np.int32)

        for node in range(n_valid_cubes, n_nodes):
            uf[node] = -1
        components = n_static_components

        # At the largest primary threshold, valid cubes or faces that exceed a
        # secondary threshold are already in the complement.
        for a in range(n0):
            for q in range(n1):
                base = (a * n1 + q) * n2
                for r in range(n2):
                    if q <= j and r <= k:
                        continue
                    bucket = base + r
                    for pos in range(cube_offsets[bucket], cube_offsets[bucket + 1]):
                        node = cube_order[pos]
                        uf[node] = -1
                        components += 1

        for a in range(n0):
            for q in range(n1):
                base = (a * n1 + q) * n2
                for r in range(n2):
                    if q <= j and r <= k:
                        continue
                    bucket = base + r
                    for pos in range(face_offsets[bucket], face_offsets[bucket + 1]):
                        face = face_order[pos]
                        components -= _union(uf, face_u[face], face_v[face])

        out[n0 - 1, j, k] = components - 1

        # Reverse the primary threshold. Cubes enter the complement before the
        # corresponding faces open.
        for a in range(n0 - 2, -1, -1):
            birth = a + 1
            for q in range(j + 1):
                base = (birth * n1 + q) * n2
                for r in range(k + 1):
                    bucket = base + r
                    for pos in range(cube_offsets[bucket], cube_offsets[bucket + 1]):
                        node = cube_order[pos]
                        uf[node] = -1
                        components += 1

            for q in range(j + 1):
                base = (birth * n1 + q) * n2
                for r in range(k + 1):
                    bucket = base + r
                    for pos in range(face_offsets[bucket], face_offsets[bucket + 1]):
                        face = face_order[pos]
                        components -= _union(uf, face_u[face], face_v[face])
            out[a, j, k] = components - 1
    return out
