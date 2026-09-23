"""Direct compact birth-event generation from top-cube bins.

This avoids materializing full birth arrays for all eight cubical-cell families.
Only cells that belong to the ROI-induced cubical complex are emitted.
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _valid(mask, x, y, z):
    X, Y, Z = mask.shape
    return 0 <= x < X and 0 <= y < Y and 0 <= z < Z and mask[x, y, z] != 0


@njit(cache=True)
def compact_primal_births(bins, mask, n_vertices, n_edges):
    X, Y, Z = mask.shape
    vb = np.empty((n_vertices, 3), dtype=bins.dtype)
    eb = np.empty((n_edges, 3), dtype=bins.dtype)
    pv = 0

    # Vertices: family order is the natural vertex lattice order.
    for x in range(X + 1):
        for y in range(Y + 1):
            for z in range(Z + 1):
                found = False
                m0 = 0; m1 = 0; m2 = 0
                for dx in range(2):
                    cx = x - dx
                    if cx < 0 or cx >= X: continue
                    for dy in range(2):
                        cy = y - dy
                        if cy < 0 or cy >= Y: continue
                        for dz in range(2):
                            cz = z - dz
                            if cz < 0 or cz >= Z or mask[cx, cy, cz] == 0: continue
                            a0 = int(bins[cx, cy, cz, 0]); a1 = int(bins[cx, cy, cz, 1]); a2 = int(bins[cx, cy, cz, 2])
                            if not found:
                                m0 = a0; m1 = a1; m2 = a2; found = True
                            else:
                                if a0 < m0: m0 = a0
                                if a1 < m1: m1 = a1
                                if a2 < m2: m2 = a2
                if found:
                    vb[pv, 0] = m0; vb[pv, 1] = m1; vb[pv, 2] = m2; pv += 1

    pe = 0
    # x edges: x cube coordinate fixed; y/z may be on either side.
    for x in range(X):
        for y in range(Y + 1):
            for z in range(Z + 1):
                found = False; m0 = 0; m1 = 0; m2 = 0
                for dy in range(2):
                    cy = y - dy
                    if cy < 0 or cy >= Y: continue
                    for dz in range(2):
                        cz = z - dz
                        if cz < 0 or cz >= Z or mask[x, cy, cz] == 0: continue
                        a0 = int(bins[x, cy, cz, 0]); a1 = int(bins[x, cy, cz, 1]); a2 = int(bins[x, cy, cz, 2])
                        if not found: m0 = a0; m1 = a1; m2 = a2; found = True
                        else:
                            if a0 < m0: m0 = a0
                            if a1 < m1: m1 = a1
                            if a2 < m2: m2 = a2
                if found:
                    eb[pe, 0] = m0; eb[pe, 1] = m1; eb[pe, 2] = m2; pe += 1

    # y edges.
    for x in range(X + 1):
        for y in range(Y):
            for z in range(Z + 1):
                found = False; m0 = 0; m1 = 0; m2 = 0
                for dx in range(2):
                    cx = x - dx
                    if cx < 0 or cx >= X: continue
                    for dz in range(2):
                        cz = z - dz
                        if cz < 0 or cz >= Z or mask[cx, y, cz] == 0: continue
                        a0 = int(bins[cx, y, cz, 0]); a1 = int(bins[cx, y, cz, 1]); a2 = int(bins[cx, y, cz, 2])
                        if not found: m0 = a0; m1 = a1; m2 = a2; found = True
                        else:
                            if a0 < m0: m0 = a0
                            if a1 < m1: m1 = a1
                            if a2 < m2: m2 = a2
                if found:
                    eb[pe, 0] = m0; eb[pe, 1] = m1; eb[pe, 2] = m2; pe += 1

    # z edges.
    for x in range(X + 1):
        for y in range(Y + 1):
            for z in range(Z):
                found = False; m0 = 0; m1 = 0; m2 = 0
                for dx in range(2):
                    cx = x - dx
                    if cx < 0 or cx >= X: continue
                    for dy in range(2):
                        cy = y - dy
                        if cy < 0 or cy >= Y or mask[cx, cy, z] == 0: continue
                        a0 = int(bins[cx, cy, z, 0]); a1 = int(bins[cx, cy, z, 1]); a2 = int(bins[cx, cy, z, 2])
                        if not found: m0 = a0; m1 = a1; m2 = a2; found = True
                        else:
                            if a0 < m0: m0 = a0
                            if a1 < m1: m1 = a1
                            if a2 < m2: m2 = a2
                if found:
                    eb[pe, 0] = m0; eb[pe, 1] = m1; eb[pe, 2] = m2; pe += 1

    if pv != n_vertices or pe != n_edges:
        raise RuntimeError("prepared primal connectivity count mismatch")
    return vb, eb


@njit(cache=True)
def compact_dual_births(bins, mask, n_valid_cubes, n_faces):
    X, Y, Z = mask.shape
    cb = np.empty((n_valid_cubes, 3), dtype=bins.dtype)
    fb = np.empty((n_faces, 3), dtype=bins.dtype)

    pc = 0
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if mask[x, y, z] != 0:
                    cb[pc, 0] = bins[x, y, z, 0]
                    cb[pc, 1] = bins[x, y, z, 1]
                    cb[pc, 2] = bins[x, y, z, 2]
                    pc += 1

    pf = 0
    # yz faces, normal to x.
    for x in range(X + 1):
        for y in range(Y):
            for z in range(Z):
                lv = x > 0 and mask[x - 1, y, z] != 0
                rv = x < X and mask[x, y, z] != 0
                if not (lv or rv): continue
                if lv:
                    m0 = int(bins[x - 1, y, z, 0]); m1 = int(bins[x - 1, y, z, 1]); m2 = int(bins[x - 1, y, z, 2])
                    if rv:
                        a0 = int(bins[x, y, z, 0]); a1 = int(bins[x, y, z, 1]); a2 = int(bins[x, y, z, 2])
                        if a0 < m0: m0 = a0
                        if a1 < m1: m1 = a1
                        if a2 < m2: m2 = a2
                else:
                    m0 = int(bins[x, y, z, 0]); m1 = int(bins[x, y, z, 1]); m2 = int(bins[x, y, z, 2])
                fb[pf, 0] = m0; fb[pf, 1] = m1; fb[pf, 2] = m2; pf += 1

    # xz faces, normal to y.
    for x in range(X):
        for y in range(Y + 1):
            for z in range(Z):
                lv = y > 0 and mask[x, y - 1, z] != 0
                rv = y < Y and mask[x, y, z] != 0
                if not (lv or rv): continue
                if lv:
                    m0 = int(bins[x, y - 1, z, 0]); m1 = int(bins[x, y - 1, z, 1]); m2 = int(bins[x, y - 1, z, 2])
                    if rv:
                        a0 = int(bins[x, y, z, 0]); a1 = int(bins[x, y, z, 1]); a2 = int(bins[x, y, z, 2])
                        if a0 < m0: m0 = a0
                        if a1 < m1: m1 = a1
                        if a2 < m2: m2 = a2
                else:
                    m0 = int(bins[x, y, z, 0]); m1 = int(bins[x, y, z, 1]); m2 = int(bins[x, y, z, 2])
                fb[pf, 0] = m0; fb[pf, 1] = m1; fb[pf, 2] = m2; pf += 1

    # xy faces, normal to z.
    for x in range(X):
        for y in range(Y):
            for z in range(Z + 1):
                lv = z > 0 and mask[x, y, z - 1] != 0
                rv = z < Z and mask[x, y, z] != 0
                if not (lv or rv): continue
                if lv:
                    m0 = int(bins[x, y, z - 1, 0]); m1 = int(bins[x, y, z - 1, 1]); m2 = int(bins[x, y, z - 1, 2])
                    if rv:
                        a0 = int(bins[x, y, z, 0]); a1 = int(bins[x, y, z, 1]); a2 = int(bins[x, y, z, 2])
                        if a0 < m0: m0 = a0
                        if a1 < m1: m1 = a1
                        if a2 < m2: m2 = a2
                else:
                    m0 = int(bins[x, y, z, 0]); m1 = int(bins[x, y, z, 1]); m2 = int(bins[x, y, z, 2])
                fb[pf, 0] = m0; fb[pf, 1] = m1; fb[pf, 2] = m2; pf += 1

    if pc != n_valid_cubes or pf != n_faces:
        raise RuntimeError("prepared dual connectivity count mismatch")
    return cb, fb


@njit(cache=True)
def add_birth_histogram(delta, births, sign):
    for i in range(births.shape[0]):
        a = int(births[i, 0]); b = int(births[i, 1]); c = int(births[i, 2])
        delta[a, b, c] += sign
