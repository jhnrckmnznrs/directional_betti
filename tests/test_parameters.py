import numpy as np
import pytest

from directional_betti import frame_invariant_directional_betti


def run(**overrides):
    kwargs = dict(covariance_radius=1.0, cone_radius=1.0, cone_half_extent=1.0,
                  directions=("z",))
    kwargs.update(overrides)
    return frame_invariant_directional_betti(np.ones((2, 2, 2)), **kwargs)


def test_fractional_grid_steps_are_rejected():
    with pytest.raises(ValueError, match="three positive integers"):
        run(grid_steps=(2.5, 2, 2))


@pytest.mark.parametrize("parameter", [
    "spacing", "covariance_radius", "cone_radius", "cone_half_extent",
    "covariance_sigma", "truncate_sigma",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_geometry_is_rejected(parameter, value):
    with pytest.raises(ValueError, match="finite"):
        run(**{parameter: value})
