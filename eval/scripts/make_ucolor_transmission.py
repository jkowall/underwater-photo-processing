#!/usr/bin/env python3
"""Build DA-V2 → transmission maps for PyTorch Ucolor inference.

Recipe (matches bakeoff TF1 path):
  z_norm = (z - z.min) / (z.max - z.min + eps)
  t = exp(-beta * z_norm)
  save as 3-channel uint8 PNG (t * 255), same HxW as depth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def depth_to_transmission(depth: np.ndarray, beta: float = 1.0) -> np.ndarray:
    z = depth.astype(np.float32)
    zmin, zmax = float(z.min()), float(z.max())
    z_norm = (z - zmin) / (zmax - zmin + 1e-8)
    t = np.exp(-beta * z_norm).astype(np.float32)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--depth-dir",
        type=Path,
        default=Path(
            "/mnt/c/Users/jkowa/underwater-photo-processing/eval/out/deepsee/depth_vis"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/mnt/c/Users/jkowa/underwater-photo-processing/eval/out/ucolor_pt/trans"
        ),
    )
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_files = sorted(args.depth_dir.glob("*_raw_depth_meter.npy"))
    if not npy_files:
        raise SystemExit(f"no depth npy in {args.depth_dir}")

    for npy in npy_files:
        stem = npy.name.replace("_raw_depth_meter.npy", "")
        depth = np.load(npy)
        t = depth_to_transmission(depth, beta=args.beta)
        rgb = np.stack([t, t, t], axis=-1)
        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        out = args.out_dir / f"{stem}.png"
        Image.fromarray(rgb_u8).save(out)
        print(
            f"OK {stem} depth=[{depth.min():.2f},{depth.max():.2f}] "
            f"t=[{t.min():.3f},{t.max():.3f}] -> {out}"
        )


if __name__ == "__main__":
    main()
