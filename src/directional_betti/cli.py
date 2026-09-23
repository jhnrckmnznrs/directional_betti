"""Command-line interface for directional-betti."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

from .api import cubical_betti_tensors, frame_invariant_directional_betti
from .io import read_volume


def _configure_threads(n: int | None) -> None:
    if n is None:
        return
    if n <= 0:
        raise ValueError("--threads must be positive")
    from numba import set_num_threads
    set_num_threads(int(n))


def _save_tensors(output: Path, result) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "betti0.npy", result.b0)
    np.save(output / "betti1.npy", result.b1)
    np.save(output / "betti2.npy", result.b2)
    np.save(output / "euler.npy", result.euler)
    (output / "grid.json").write_text(json.dumps(asdict(result.grid), indent=2) + "\n")


def _topology(args: argparse.Namespace) -> None:
    filtration = read_volume(args.filtration, allow_4d=True)
    roi = None if args.roi is None else read_volume(args.roi)
    result = cubical_betti_tensors(
        filtration,
        roi,
        grid_steps=(args.grid_res,) * 3,
    )
    _save_tensors(args.output, result)
    print(f"saved tensors {result.b0.shape} -> {args.output}")


def _invariant(args: argparse.Namespace) -> None:
    foreground = read_volume(args.foreground)
    roi = None if args.roi is None else read_volume(args.roi)
    analysis = frame_invariant_directional_betti(
        foreground,
        roi,
        spacing=args.spacing,
        directions=args.directions,
        covariance_radius=args.covariance_radius,
        cone_radius=args.cone_radius,
        cone_half_extent=args.cone_half_extent,
        covariance_kernel=args.covariance_kernel,
        covariance_sigma=args.covariance_sigma,
        truncate_sigma=args.truncate_sigma,
        grid_steps=(args.grid_res,) * 3,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    grids = {}
    for direction, result in analysis.by_direction.items():
        direction_dir = args.output / direction
        _save_tensors(direction_dir, result)
        grids[direction] = asdict(result.grid)

    config = {
        "spacing": analysis.spacing,
        "directions": list(analysis.by_direction),
        "covariance_radius": analysis.covariance_radius,
        "cone_radius": analysis.cone_radius,
        "cone_half_extent": analysis.cone_half_extent,
        "covariance_kernel": analysis.covariance_kernel,
        "covariance_sigma": args.covariance_sigma,
        "truncate_sigma": args.truncate_sigma,
        "grid_res": args.grid_res,
    }
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"saved directions {','.join(analysis.by_direction)} -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="directional-betti",
        description="Exact sampled cubical Betti tensors for 3-D voxel data.",
    )
    parser.add_argument("--threads", type=int, help="Numba thread count")
    subparsers = parser.add_subparsers(dest="command", required=True)

    topology = subparsers.add_parser(
        "topology",
        help="compute tensors from a saved three-parameter filtration",
    )
    topology.add_argument("filtration", type=Path, help=".npy array with shape (X,Y,Z,3)")
    topology.add_argument("output", type=Path)
    topology.add_argument("--roi", type=Path, help="optional 3-D ROI mask")
    topology.add_argument("--grid-res", type=int, default=8)
    topology.set_defaults(func=_topology)

    invariant = subparsers.add_parser(
        "invariant",
        help="compute frame-invariant (C_d,A_d,S) tensors from a binary volume",
    )
    invariant.add_argument("foreground", type=Path, help="3-D .npy or TIFF volume")
    invariant.add_argument("output", type=Path)
    invariant.add_argument("--roi", type=Path, help="optional 3-D .npy or TIFF ROI mask")
    invariant.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="isotropic voxel spacing in a consistent physical unit",
    )
    invariant.add_argument(
        "--directions",
        nargs="+",
        choices=["x", "y", "z"],
        default=["x", "y", "z"],
    )
    invariant.add_argument("--covariance-radius", type=float, required=True)
    invariant.add_argument("--cone-radius", type=float, required=True)
    invariant.add_argument("--cone-half-extent", type=float, required=True)
    invariant.add_argument(
        "--covariance-kernel",
        choices=["hard", "gaussian"],
        default="gaussian",
    )
    invariant.add_argument("--covariance-sigma", type=float)
    invariant.add_argument("--truncate-sigma", type=float, default=4.0)
    invariant.add_argument("--grid-res", type=int, default=8)
    invariant.set_defaults(func=_invariant)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        _configure_threads(args.threads)
        args.func(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
