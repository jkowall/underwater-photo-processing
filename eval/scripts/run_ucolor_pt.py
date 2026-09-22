#!/usr/bin/env python3
"""Run PyTorch Ucolor on bakeoff stills with precomputed transmission maps.

Uses overlapping tiles so 1536×1024 fits in 16GB VRAM (model is ~105M params).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

REPO = Path(__file__).resolve().parents[1] / "repos" / "ucolor_pytorch"
sys.path.insert(0, str(REPO))

from models import Model  # noqa: E402
from utils import load_checkpoint  # noqa: E402


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    im = Image.open(path).convert("RGB")
    return TF.to_tensor(im).unsqueeze(0).to(device)


def infer_tiled(
    model: torch.nn.Module,
    inp: torch.Tensor,
    dep: torch.Tensor,
    tile: int = 512,
    overlap: int = 64,
) -> torch.Tensor:
    """inp/dep: 1x3xHxW in [0,1]. Returns 1x3xHxW."""
    _, _, h, w = inp.shape
    if h <= tile and w <= tile:
        with torch.no_grad():
            return model(inp, dep).clamp(0, 1)

    step = tile - overlap
    out = torch.zeros_like(inp)
    weight = torch.zeros((1, 1, h, w), device=inp.device, dtype=inp.dtype)

    # raised-cosine-ish blend weights inside each tile
    yy = torch.linspace(0, 1, tile, device=inp.device)
    xx = torch.linspace(0, 1, tile, device=inp.device)
    wy = 0.5 - 0.5 * torch.cos(torch.pi * yy.clamp(0, 1))
    wx = 0.5 - 0.5 * torch.cos(torch.pi * xx.clamp(0, 1))
    base_w = (wy[:, None] * wx[None, :]).view(1, 1, tile, tile)

    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch_i = inp[:, :, y : y + tile, x : x + tile]
                patch_d = dep[:, :, y : y + tile, x : x + tile]
                pred = model(patch_i, patch_d).clamp(0, 1)
                th, tw = pred.shape[-2:]
                wgt = base_w[:, :, :th, :tw]
                out[:, :, y : y + th, x : x + tw] += pred * wgt
                weight[:, :, y : y + th, x : x + tw] += wgt

    return out / weight.clamp_min(1e-8)


def percentile_stretch(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr.astype(np.float32), [1, 99])
    if hi <= lo:
        return arr
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(
        np.uint8
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=REPO / "checkpoints" / "best.pth")
    ap.add_argument(
        "--stills",
        type=Path,
        default=Path("/mnt/c/Users/jkowa/underwater-photo-processing/eval/stills"),
    )
    ap.add_argument(
        "--trans",
        type=Path,
        default=Path(
            "/mnt/c/Users/jkowa/underwater-photo-processing/eval/out/ucolor_pt/trans"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/c/Users/jkowa/underwater-photo-processing/eval/out/ucolor_pt"),
    )
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"missing weights: {args.weights}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    model = Model().to(device)
    load_checkpoint(model, str(args.weights))
    model.eval()

    stills = sorted(args.stills.glob("*.jpg"))
    if not stills:
        raise SystemExit(f"no stills in {args.stills}")

    t0 = time.time()
    peak = 0.0
    ok = 0
    for still in stills:
        stem = still.stem
        trans_path = args.trans / f"{stem}.png"
        if not trans_path.exists():
            print("MISSING trans", trans_path)
            continue
        inp = load_rgb_tensor(still, device)
        dep = load_rgb_tensor(trans_path, device)
        if dep.shape[-2:] != inp.shape[-2:]:
            dep = torch.nn.functional.interpolate(
                dep, size=inp.shape[-2:], mode="bilinear", align_corners=False
            )

        out = infer_tiled(model, inp, dep, tile=args.tile, overlap=args.overlap)
        arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        arr = percentile_stretch(arr)
        Image.fromarray(arr).save(args.out / f"{stem}.png")
        ok += 1
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.reset_peak_memory_stats()
        print("OK", stem, arr.shape)

    wall = time.time() - t0
    (args.out / "infer.log").write_text(
        f"STATUS=SUCCESS\nn={ok}\nwall_s={wall:.2f}\n"
        f"weights={args.weights}\ndevice={device}\n"
        f"tile={args.tile}\noverlap={args.overlap}\npeak_mem_gb={peak:.2f}\n"
    )
    print(f"done n={ok} wall_s={wall:.2f} peak_gb={peak:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
