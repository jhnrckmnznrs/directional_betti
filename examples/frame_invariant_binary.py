"""Frame-invariant directional topology on a synthetic binary object."""
import numpy as np
from directional_betti import frame_invariant_directional_betti

obj = np.zeros((32, 32, 32), dtype=np.uint8)
obj[6:26, 15:18, 15:18] = 1
obj[15:18, 6:26, 15:18] = 1
roi = np.ones_like(obj)

analysis = frame_invariant_directional_betti(
    obj,
    roi,
    spacing=1.0,
    directions=("x", "y", "z"),
    covariance_radius=4.0,
    cone_radius=4.0,
    cone_half_extent=8.0,
    covariance_kernel="gaussian",
)

for d, result in analysis.by_direction.items():
    result.assert_consistent()
    print(d, "B0 terminal =", int(result.b0[-1, -1, -1]))
