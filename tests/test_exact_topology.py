"""Independent F_2 boundary-matrix checks of the sampled cubical complexes.

The oracle deliberately does not use production cell events, connectivity,
Alexander duality, or the Euler identity to compute Betti numbers.
"""
from itertools import product

import numpy as np
import pytest

from directional_betti import cubical_betti_tensors
from directional_betti.topology import betti_tensors_from_bins


def explicit_cells(bins, roi):
    """Enumerate all faces and take incident top-cube minima coordinatewise.

    A cell is a tuple of three (lower endpoint, extent) pairs; extent is 0 or 1.
    """
    births = {}
    for index in np.ndindex(roi.shape):
        if not roi[index]:
            continue
        for kind in product((0, 1, 2), repeat=3):
            cell = tuple((x + (k == 2), int(k == 1))
                         for x, k in zip(index, kind))
            value = tuple(int(v) for v in bins[index])
            if cell in births:
                value = tuple(min(a, b) for a, b in zip(value, births[cell]))
            births[cell] = value
    return births


def boundary(cell):
    for axis, (start, extent) in enumerate(cell):
        if extent:
            for offset in (0, 1):
                face = list(cell)
                face[axis] = (start + offset, 0)
                yield tuple(face)


def rank_mod_two(columns):
    """Gaussian elimination on bit-encoded columns, without floating arithmetic."""
    pivots = {}
    for column in columns:
        while column:
            pivot = column.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = column
                break
            column ^= pivots[pivot]
    return len(pivots)


def explicit_homology(births, threshold):
    cells = [[] for _ in range(4)]
    for cell, birth in births.items():
        if all(b <= t for b, t in zip(birth, threshold)):
            cells[sum(extent for _, extent in cell)].append(cell)
    ranks = [0]
    for dimension in range(1, 4):
        rows = {cell: i for i, cell in enumerate(cells[dimension - 1])}
        columns = []
        for cell in cells[dimension]:
            column = 0
            for face in boundary(cell):
                column ^= 1 << rows[face]  # Also verifies closure under faces.
            columns.append(column)
        ranks.append(rank_mod_two(columns))
    ranks.append(0)
    betti = tuple(len(cells[q]) - ranks[q] - ranks[q + 1] for q in range(4))
    chi = sum((-1) ** q * len(cells[q]) for q in range(4))
    return betti, chi


@pytest.mark.parametrize("seed", range(12))
def test_random_multifiltrations_against_boundary_matrices(seed):
    rng = np.random.default_rng(seed)
    shape = ((2, 3, 4), (4, 2, 3), (3, 4, 2))[seed % 3]
    # Exercise all three choices of sweep axis, including unequal grid sizes.
    output_shape = ((4, 2, 3), (2, 4, 3), (2, 3, 4))[seed % 3]
    roi = rng.random(shape) > 0.35
    bins = rng.integers(0, output_shape, size=shape + (3,), dtype=np.int16)
    bins[~roi] = -1
    actual = betti_tensors_from_bins(bins, roi, output_shape)
    births = explicit_cells(bins, roi)
    for threshold in np.ndindex(output_shape):
        betti, chi = explicit_homology(births, threshold)
        assert betti[3] == 0
        assert tuple(int(t[threshold]) for t in actual) == (*betti[:3], chi)


@pytest.mark.parametrize("phantom, expected", [
    ("cube", (1, 0, 0)),
    ("disconnected", (2, 0, 0)),
    ("tunnel", (1, 1, 0)),
    ("cavity", (1, 0, 1)),
])
def test_terminal_roi_topology(phantom, expected):
    roi = np.ones((3, 3, 3), dtype=bool)
    if phantom == "disconnected":
        roi[1] = False
    elif phantom == "tunnel":
        roi[1, 1, :] = False
    elif phantom == "cavity":
        roi[1, 1, 1] = False
    filtration = np.zeros(roi.shape + (3,), dtype=np.float64)
    result = cubical_betti_tensors(filtration, roi, grid_steps=(2, 2, 2))
    for actual, value in zip((result.b0, result.b1, result.b2), expected):
        np.testing.assert_array_equal(actual, np.full((3, 3, 3), value))


def test_face_can_enter_before_any_incident_cube():
    # Adjacent cubes satisfy different coordinates. At (0,0,0), their shared
    # square and its boundary enter, while neither cube enters.
    bins = np.array([[[[0, 1, 0]]], [[[1, 0, 0]]]], dtype=np.int8)
    roi = np.ones((2, 1, 1), dtype=bool)
    tensors = betti_tensors_from_bins(bins, roi, (2, 2, 1))
    betti, chi = explicit_homology(explicit_cells(bins, roi), (0, 0, 0))
    assert (*betti[:3], chi) == (1, 0, 0, 1)
    assert tuple(int(t[0, 0, 0]) for t in tensors) == (1, 0, 0, 1)
