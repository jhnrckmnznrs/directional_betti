import numpy as np

from directional_betti.grid import bin_multifiltration


def test_cumulative_bins_include_lower_edge_and_exclude_upper_edge():
    # Exactly representable edges avoid conflating semantics with roundoff.
    values = np.linspace(-1.0, 1.0, 9)
    filtration = np.broadcast_to(values[:, None, None, None], (9, 1, 1, 3)).copy()
    bins, _, spec = bin_multifiltration(filtration)
    np.testing.assert_array_equal(bins[:, 0, 0, 0], np.arange(9))
    assert spec.step_sizes == (0.25, 0.25, 0.25)
    for a in range(8):
        np.testing.assert_array_equal(bins[:, 0, 0, 0] <= a,
                                      values < -1.0 + (a + 1) * 0.25)


def test_constant_coordinates_and_values_outside_roi():
    filtration = np.full((2, 2, 2, 3), 4.0)
    roi = np.ones((2, 2, 2), dtype=bool)
    roi[0, 0, 0] = False
    filtration[0, 0, 0] = np.nan
    bins, _, spec = bin_multifiltration(filtration, roi)
    np.testing.assert_array_equal(bins[roi], np.zeros((7, 3), dtype=np.int8))
    np.testing.assert_array_equal(bins[~roi], -np.ones((1, 3), dtype=np.int8))
    assert spec.minima == spec.maxima == (4.0, 4.0, 4.0)
