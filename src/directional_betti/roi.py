"""Reusable compact cubical connectivity for exact directional Betti sweeps."""
from __future__ import annotations
from dataclasses import dataclass
import zlib
import numpy as np
from numba import njit


@dataclass(frozen=True)
class PreparedCubicalROI:
    """Cubical graph connectivity that depends only on the top-cube ROI mask.

    Valid primal vertices are compact nodes ``0..n_vertices-1``.  ``edge_u``
    and ``edge_v`` store endpoints of every valid cubical edge in family order
    x, y, z.

    In the Alexander-dual graph, valid cubes are dynamic nodes
    ``0..n_valid_cubes-1``.  Invalid cubes and the distinguished exterior are
    contracted through permanently absent faces; the resulting static nodes
    occupy the remaining node ids.  ``face_u`` and ``face_v`` contain only
    faces incident to at least one valid cube, in family order yz, xz, xy.
    """
    shape: tuple[int, int, int]
    mask_crc32: int
    n_vertices: int
    edge_u: np.ndarray
    edge_v: np.ndarray
    n_valid_cubes: int
    n_static_components: int
    n_nodes: int
    exterior_node: int
    face_u: np.ndarray
    face_v: np.ndarray


def _crc32_mask(mask: np.ndarray) -> int:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1))
    return int(zlib.crc32(memoryview(packed)) & 0xFFFFFFFF)


@njit(cache=True, inline="always")
def _uf_find(uf, a):
    root = a
    while uf[root] >= 0:
        root = uf[root]
    while a != root and uf[a] >= 0:
        nxt = uf[a]
        uf[a] = root
        a = nxt
    return root


@njit(cache=True, inline="always")
def _uf_union(uf, a, b):
    ra = _uf_find(uf, a)
    rb = _uf_find(uf, b)
    if ra == rb:
        return
    if uf[ra] > uf[rb]:
        ra, rb = rb, ra
    uf[ra] += uf[rb]
    uf[rb] = ra


@njit(cache=True, inline="always")
def _cube_valid(mask, x, y, z):
    X, Y, Z = mask.shape
    return 0 <= x < X and 0 <= y < Y and 0 <= z < Z and mask[x, y, z] != 0


@njit(cache=True)
def _prepare_connectivity(mask):
    X, Y, Z = mask.shape
    Nc = X * Y * Z

    # ---------- compact primal vertices ----------
    vmap = np.full((X + 1, Y + 1, Z + 1), -1, dtype=np.int32)
    nvtx = 0
    for x in range(X + 1):
        for y in range(Y + 1):
            for z in range(Z + 1):
                present = False
                for dx in range(2):
                    for dy in range(2):
                        for dz in range(2):
                            if _cube_valid(mask, x - dx, y - dy, z - dz):
                                present = True
                if present:
                    vmap[x, y, z] = nvtx
                    nvtx += 1

    # Count valid edges (x, y, z family order).
    ne = 0
    for x in range(X):
        for y in range(Y + 1):
            for z in range(Z + 1):
                present = False
                for dy in range(2):
                    for dz in range(2):
                        if _cube_valid(mask, x, y - dy, z - dz): present = True
                if present: ne += 1
    for x in range(X + 1):
        for y in range(Y):
            for z in range(Z + 1):
                present = False
                for dx in range(2):
                    for dz in range(2):
                        if _cube_valid(mask, x - dx, y, z - dz): present = True
                if present: ne += 1
    for x in range(X + 1):
        for y in range(Y + 1):
            for z in range(Z):
                present = False
                for dx in range(2):
                    for dy in range(2):
                        if _cube_valid(mask, x - dx, y - dy, z): present = True
                if present: ne += 1

    eu = np.empty(ne, dtype=np.int32)
    ev = np.empty(ne, dtype=np.int32)
    p = 0
    for x in range(X):
        for y in range(Y + 1):
            for z in range(Z + 1):
                present = False
                for dy in range(2):
                    for dz in range(2):
                        if _cube_valid(mask, x, y - dy, z - dz): present = True
                if present:
                    eu[p] = vmap[x, y, z]; ev[p] = vmap[x + 1, y, z]; p += 1
    for x in range(X + 1):
        for y in range(Y):
            for z in range(Z + 1):
                present = False
                for dx in range(2):
                    for dz in range(2):
                        if _cube_valid(mask, x - dx, y, z - dz): present = True
                if present:
                    eu[p] = vmap[x, y, z]; ev[p] = vmap[x, y + 1, z]; p += 1
    for x in range(X + 1):
        for y in range(Y + 1):
            for z in range(Z):
                present = False
                for dx in range(2):
                    for dy in range(2):
                        if _cube_valid(mask, x - dx, y - dy, z): present = True
                if present:
                    eu[p] = vmap[x, y, z]; ev[p] = vmap[x, y, z + 1]; p += 1

    # ---------- contracted dual complement ----------
    nvc = 0
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if mask[x, y, z] != 0: nvc += 1

    node_map = np.empty(Nc, dtype=np.int32)
    uf = np.empty(Nc + 1, dtype=np.int32)
    ext_old = Nc
    uf[ext_old] = -1
    iv = 0
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                old = (x * Y + y) * Z + z
                if mask[x, y, z] != 0:
                    node_map[old] = iv; iv += 1
                else:
                    uf[old] = -1

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if mask[x, y, z] != 0: continue
                old = (x * Y + y) * Z + z
                if x > 0 and mask[x - 1, y, z] == 0: _uf_union(uf, old, old - Y * Z)
                if y > 0 and mask[x, y - 1, z] == 0: _uf_union(uf, old, old - Z)
                if z > 0 and mask[x, y, z - 1] == 0: _uf_union(uf, old, old - 1)
                if x == 0 or x == X - 1 or y == 0 or y == Y - 1 or z == 0 or z == Z - 1:
                    _uf_union(uf, old, ext_old)

    root_to_node = np.full(Nc + 1, -1, dtype=np.int32)
    ext_root = _uf_find(uf, ext_old)
    root_to_node[ext_root] = nvc
    next_node = nvc + 1
    for old in range(Nc):
        x = old // (Y * Z)
        rem = old - x * Y * Z
        y = rem // Z
        z = rem - y * Z
        if mask[x, y, z] != 0: continue
        root = _uf_find(uf, old)
        if root_to_node[root] < 0:
            root_to_node[root] = next_node; next_node += 1
        node_map[old] = root_to_node[root]
    nstatic = next_node - nvc
    ext_node = nvc

    # Count dynamic faces: incident to at least one valid cube.
    nf = 0
    for x in range(X + 1):
        for y in range(Y):
            for z in range(Z):
                if _cube_valid(mask, x - 1, y, z) or _cube_valid(mask, x, y, z): nf += 1
    for x in range(X):
        for y in range(Y + 1):
            for z in range(Z):
                if _cube_valid(mask, x, y - 1, z) or _cube_valid(mask, x, y, z): nf += 1
    for x in range(X):
        for y in range(Y):
            for z in range(Z + 1):
                if _cube_valid(mask, x, y, z - 1) or _cube_valid(mask, x, y, z): nf += 1

    fu = np.empty(nf, dtype=np.int32)
    fv = np.empty(nf, dtype=np.int32)
    p = 0
    for x in range(X + 1):
        for y in range(Y):
            for z in range(Z):
                left = _cube_valid(mask, x - 1, y, z)
                right = _cube_valid(mask, x, y, z)
                if not (left or right): continue
                if x == 0:
                    u = ext_node; v = node_map[y * Z + z]
                elif x == X:
                    u = node_map[((X - 1) * Y + y) * Z + z]; v = ext_node
                else:
                    u = node_map[((x - 1) * Y + y) * Z + z]
                    v = node_map[(x * Y + y) * Z + z]
                fu[p] = u; fv[p] = v; p += 1
    for x in range(X):
        for y in range(Y + 1):
            for z in range(Z):
                lo = _cube_valid(mask, x, y - 1, z)
                hi = _cube_valid(mask, x, y, z)
                if not (lo or hi): continue
                if y == 0:
                    u = ext_node; v = node_map[(x * Y) * Z + z]
                elif y == Y:
                    u = node_map[(x * Y + (Y - 1)) * Z + z]; v = ext_node
                else:
                    u = node_map[(x * Y + (y - 1)) * Z + z]
                    v = node_map[(x * Y + y) * Z + z]
                fu[p] = u; fv[p] = v; p += 1
    for x in range(X):
        for y in range(Y):
            for z in range(Z + 1):
                lo = _cube_valid(mask, x, y, z - 1)
                hi = _cube_valid(mask, x, y, z)
                if not (lo or hi): continue
                if z == 0:
                    u = ext_node; v = node_map[(x * Y + y) * Z]
                elif z == Z:
                    u = node_map[(x * Y + y) * Z + (Z - 1)]; v = ext_node
                else:
                    u = node_map[(x * Y + y) * Z + (z - 1)]
                    v = node_map[(x * Y + y) * Z + z]
                fu[p] = u; fv[p] = v; p += 1

    return nvtx, eu, ev, nvc, nstatic, next_node, ext_node, fu, fv


def prepare_cubical_roi(valid_mask: np.ndarray) -> PreparedCubicalROI:
    mask = np.asarray(valid_mask)
    if mask.ndim != 3:
        raise ValueError(f"valid_mask must be 3-D, got {mask.shape}")
    mask_u8 = np.ascontiguousarray(mask != 0, dtype=np.uint8)
    nvtx, eu, ev, nvc, nstatic, nnodes, ext, fu, fv = _prepare_connectivity(mask_u8)
    return PreparedCubicalROI(
        shape=tuple(int(v) for v in mask.shape),
        mask_crc32=_crc32_mask(mask_u8),
        n_vertices=int(nvtx),
        edge_u=eu,
        edge_v=ev,
        n_valid_cubes=int(nvc),
        n_static_components=int(nstatic),
        n_nodes=int(nnodes),
        exterior_node=int(ext),
        face_u=fu,
        face_v=fv,
    )


def validate_prepared_roi(prepared: PreparedCubicalROI, valid_mask: np.ndarray) -> None:
    mask = np.asarray(valid_mask)
    if tuple(mask.shape) != prepared.shape:
        raise ValueError(f"prepared ROI shape {prepared.shape} does not match mask {mask.shape}")
    if _crc32_mask(mask != 0) != prepared.mask_crc32:
        raise ValueError("prepared ROI was built from a different valid_mask")
