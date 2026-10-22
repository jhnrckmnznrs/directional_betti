"""Lattice-preserving transverse changes of frame, without resampling."""
import numpy as np
import pytest

from directional_betti import frame_invariant_directional_betti


@pytest.mark.parametrize("kernel", ["hard", "gaussian"])
@pytest.mark.parametrize("direction, axes", [
    ("x", (1, 2)), ("y", (0, 2)), ("z", (0, 1)),
])
def test_quarter_turn_preserves_filtration_and_betti(kernel, direction, axes):
    rng = np.random.default_rng(15)
    roi = rng.random((5, 6, 7)) > 0.2
    foreground = (rng.random(roi.shape) > 0.6) & roi
    kwargs = dict(spacing=1.0, directions=(direction,), covariance_radius=2.0,
                  cone_radius=2.0, cone_half_extent=3.0,
                  covariance_kernel=kernel, grid_steps=(2, 2, 2))
    first = frame_invariant_directional_betti(foreground, roi, **kwargs)
    turned = frame_invariant_directional_betti(
        np.rot90(foreground, axes=axes), np.rot90(roi, axes=axes), **kwargs)
    a, b = first.by_direction[direction], turned.by_direction[direction]
    assert a.grid == b.grid
    for name in ("b0", "b1", "b2", "euler"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
