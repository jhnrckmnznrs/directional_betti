"""Minimal application-independent cubical topology example."""
import numpy as np
from directional_betti import cubical_betti_tensors

X, Y, Z = 8, 7, 6
F = np.empty((X, Y, Z, 3), dtype=np.float32)
F[..., 0] = np.linspace(0.0, 1.0, X)[:, None, None]
F[..., 1] = np.linspace(0.0, 1.0, Y)[None, :, None]
F[..., 2] = np.linspace(0.0, 1.0, Z)[None, None, :]

result = cubical_betti_tensors(F, grid_steps=(8, 8, 8))
result.assert_consistent()
print("tensor shape:", result.b0.shape)
print("terminal:", result.b0[-1, -1, -1], result.b1[-1, -1, -1], result.b2[-1, -1, -1])
