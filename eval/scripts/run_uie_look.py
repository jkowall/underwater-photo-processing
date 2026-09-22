#!/usr/bin/env python3
"""Bakeoff-proven Spectroformer / NU2Net inference (run inside WSL uw_eval)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

EVAL_ROOT = Path(__file__).resolve().parents[1]
SPECTRO_ROOT = EVAL_ROOT / "repos" / "spectroformer"
NU2_ROOT = EVAL_ROOT / "repos" / "uie_benchmark"
SPECTRO_CKPT = SPECTRO_ROOT / "checkpoints" / "best.pth"
NU2_CKPT = NU2_ROOT / "checkpoints" / "UIEB" / "NU2Net.ckpt"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def long_edge_resize(im: Image.Image, long_edge: int) -> Image.Image:
    w, h = im.size
    scale = long_edge / max(w, h)
    if abs(scale - 1.0) < 1e-6:
        return im
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return im.resize((nw, nh), Image.BICUBIC)


def load_spectroformer(device: torch.device):
    if not SPECTRO_CKPT.is_file():
        raise FileNotFoundError(f"missing Spectroformer weights: {SPECTRO_CKPT}")
    # Checkpoint pickles the full module; class lives in the spectroformer repo.
    sys.path.insert(0, str(SPECTRO_ROOT))
    model = torch.load(SPECTRO_CKPT, map_location=device, weights_only=False)
    model = model.to(device).eval()
    return model


def infer_spectroformer(model, rgb: Image.Image, device: torch.device) -> Image.Image:
    # Bakeoff: 512×512 infer, Normalize(0.5), upsample to target
    target = rgb.size  # (w, h)
    work = rgb.resize((512, 512), Image.BICUBIC)
    transform = T.Compose(
        [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    inp = transform(work).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.detach()
        # model outputs in [-1, 1] (save_img convention)
        out = (out + 1.0) * 0.5
        out = out.clamp(0, 1)
        out = F.interpolate(
            out, size=(target[1], target[0]), mode="bilinear", align_corners=False
        )
    arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def load_nu2net(device: torch.device):
    if not NU2_CKPT.is_file():
        raise FileNotFoundError(f"missing NU2Net weights: {NU2_CKPT}")
    sys.path.insert(0, str(NU2_ROOT))
    from model.NU2Net import NU2Net  # noqa: WPS410

    model = NU2Net().to(device)
    ckpt = torch.load(NU2_CKPT, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        if name.startswith("model."):
            name = name[6:]
        cleaned[name] = v
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def normalize_img(img: torch.Tensor) -> torch.Tensor:
    if torch.max(img) > 1 or torch.min(img) < 0:
        im_max = torch.max(img)
        im_min = torch.min(img)
        img = (img - im_min) / (im_max - im_min + 1e-7)
    return img


def infer_nu2net(model, rgb: Image.Image, device: torch.device) -> Image.Image:
    # Bakeoff: pad dims down to multiples of 16, ToTensor, upsample back
    target = rgb.size
    w, h = target
    nw, nh = (w // 16) * 16, (h // 16) * 16
    work = rgb.resize((nw, nh), Image.BICUBIC)
    inp = T.ToTensor()(work).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = normalize_img(out).clamp(0, 1)
        out = F.interpolate(
            out, size=(target[1], target[0]), mode="bilinear", align_corners=False
        )
    arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Spectroformer / NU2Net UIE infer")
    ap.add_argument("--look", choices=["spectroformer", "nu2net"], required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--long-edge",
        type=int,
        default=1536,
        help="Resize so max(w,h)=this before enhance (0=keep source size)",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input missing: {args.input}")

    images = list_images(args.input)
    if not images:
        raise SystemExit(f"no images in {args.input}")

    out_dir = args.output
    if len(images) == 1 and out_dir.suffix.lower() in IMAGE_EXTS:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        single_out = out_dir
        out_dir = None
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        single_out = None

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    t0 = time.time()
    peak = 0.0

    if args.look == "spectroformer":
        model = load_spectroformer(device)
        infer_fn = infer_spectroformer
    else:
        model = load_nu2net(device)
        infer_fn = infer_nu2net

    for src in images:
        im = Image.open(src).convert("RGB")
        if args.long_edge and args.long_edge > 0:
            im = long_edge_resize(im, args.long_edge)
        work_size = im.size
        result = infer_fn(model, im, device)
        # Always write at the working size passed to infer (caller may upsample later)
        if result.size != work_size:
            result = result.resize(work_size, Image.BICUBIC)
        dest = single_out if single_out is not None else out_dir / f"{src.stem}.png"
        result.save(dest)
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.reset_peak_memory_stats()
        print(f"OK {src.name} -> {dest} size={result.size}", flush=True)

    wall = time.time() - t0
    print(
        f"DONE look={args.look} n={len(images)} wall_s={wall:.2f} "
        f"device={device} peak_gb={peak:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
