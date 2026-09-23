"""Small file-format helpers used by the command-line interface."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def read_volume(path: Path, *, allow_4d: bool = False) -> np.ndarray:
    """Read a NumPy or TIFF volume.

    NumPy input is supported by the core package.  TIFF input requires the
    optional ``io`` dependency.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError(
                "TIFF input requires: pip install 'directional-betti[io]'"
            ) from exc
        arr = tifffile.imread(path)
    else:
        raise ValueError(f"unsupported volume format: {path}")

    allowed = {3, 4} if allow_4d else {3}
    if arr.ndim not in allowed:
        expected = "3-D or 4-D" if allow_4d else "3-D"
        raise ValueError(f"expected {expected} array, got {arr.shape}")
    return arr
