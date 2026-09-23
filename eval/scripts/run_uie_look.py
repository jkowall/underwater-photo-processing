#!/usr/bin/env python3
"""Bakeoff-proven Spectroformer / NU2Net inference (Windows CUDA, Apple MPS, or WSL)."""
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


def pick_device(requested: str) -> torch.device:
    """cuda → mps → cpu, or honor an explicit non-auto request when available."""
    req = (requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    if req == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        raise SystemExit("MPS requested but not available")
    return torch.device(req)


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


def infer_spectroformer(
    model,
    rgb: Image.Image,
    device: torch.device,
    *,
    multiple: int = 8,
) -> Image.Image:
    """Run Spectroformer at the working resolution (aspect preserved).

    Upstream train/test used 512×512 squares. Squashing full dive frames to 512
    then upsampling to NEF size destroys sharpness. The network is fully
    convolutional with 3× downsample, so H/W only need to be multiples of 8.
    """
    target = rgb.size  # (w, h)
    w, h = target
    pw = (multiple - (w % multiple)) % multiple
    ph = (multiple - (h % multiple)) % multiple
    work = rgb
    if pw or ph:
        canvas = Image.new("RGB", (w + pw, h + ph))
        canvas.paste(rgb, (0, 0))
        work = canvas
    transform = T.Compose(
        [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    inp = transform(work).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.detach()
        out = (out + 1.0) * 0.5
        out = out.clamp(0, 1)
        if pw or ph:
            out = out[:, :, :h, :w]
        if out.shape[-1] != w or out.shape[-2] != h:
            out = F.interpolate(
                out, size=(h, w), mode="bilinear", align_corners=False
            )
    arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _tile_blend_weights(tile: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Raised-cosine weights for feathered tile blending (1x1xtilextile)."""
    yy = torch.linspace(0, 1, tile, device=device, dtype=dtype)
    xx = torch.linspace(0, 1, tile, device=device, dtype=dtype)
    wy = 0.5 - 0.5 * torch.cos(torch.pi * yy.clamp(0, 1))
    wx = 0.5 - 0.5 * torch.cos(torch.pi * xx.clamp(0, 1))
    return (wy[:, None] * wx[None, :]).view(1, 1, tile, tile)


def infer_spectroformer_tiled(
    model,
    rgb: Image.Image,
    device: torch.device,
    *,
    tile: int = 1536,
    overlap: int = 192,
    multiple: int = 8,
) -> Image.Image:
    """Overlapping Spectroformer tiles with feathered blend (VRAM-safe at high long-edge).

    Whole-frame 3072 already OOM'd on RTX 5080; tile size stays near the proven
    2048 working-res budget. Tile dims are rounded up to ``multiple`` for the net.
    """
    w, h = rgb.size
    tile = max(multiple, int(tile))
    tile = tile + (multiple - (tile % multiple)) % multiple
    overlap = max(0, min(int(overlap), tile - multiple))
    if h <= tile and w <= tile:
        return infer_spectroformer(model, rgb, device, multiple=multiple)

    transform = T.Compose(
        [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    inp = transform(rgb).unsqueeze(0).to(device)
    out = torch.zeros((1, 3, h, w), device=device, dtype=inp.dtype)
    weight = torch.zeros((1, 1, h, w), device=device, dtype=inp.dtype)
    base_w = _tile_blend_weights(tile, device, inp.dtype)

    step = max(multiple, tile - overlap)
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if not ys or ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    if not xs or xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    # de-dupe while preserving order
    ys = list(dict.fromkeys(ys))
    xs = list(dict.fromkeys(xs))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                y1, x1 = min(y + tile, h), min(x + tile, w)
                y0, x0 = y1 - tile, x1 - tile
                if y0 < 0:
                    y0, y1 = 0, min(tile, h)
                if x0 < 0:
                    x0, x1 = 0, min(tile, w)
                patch = inp[:, :, y0:y1, x0:x1]
                ph, pw = patch.shape[-2], patch.shape[-1]
                pad_h = (multiple - (ph % multiple)) % multiple
                pad_w = (multiple - (pw % multiple)) % multiple
                if pad_h or pad_w:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="replicate")
                pred = model(patch)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred = ((pred + 1.0) * 0.5).clamp(0, 1)[:, :, :ph, :pw]
                wgt = base_w[:, :, :ph, :pw]
                out[:, :, y0:y1, x0:x1] += pred * wgt
                weight[:, :, y0:y1, x0:x1] += wgt

    merged = out / weight.clamp_min(1e-8)
    arr = (merged.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
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
        default=2048,
        help="Resize so max(w,h)=this before enhance (0=keep source size)",
    )
    ap.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Spectroformer tile size (0=whole frame). Use with high --long-edge.",
    )
    ap.add_argument(
        "--overlap",
        type=int,
        default=192,
        help="Overlap pixels between Spectroformer tiles when --tile > 0",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="auto (cuda→mps→cpu), or cuda / mps / cpu",
    )
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

    device = pick_device(args.device)
    t0 = time.time()
    peak = 0.0

    if args.look == "spectroformer":
        model = load_spectroformer(device)
    else:
        model = load_nu2net(device)

    for src in images:
        im = Image.open(src).convert("RGB")
        if args.long_edge and args.long_edge > 0:
            im = long_edge_resize(im, args.long_edge)
        work_size = im.size
        if args.look == "spectroformer" and args.tile and args.tile > 0:
            result = infer_spectroformer_tiled(
                model, im, device, tile=args.tile, overlap=args.overlap
            )
        elif args.look == "spectroformer":
            result = infer_spectroformer(model, im, device)
        else:
            result = infer_nu2net(model, im, device)
        # Always write at the working size passed to infer (caller may upsample later)
        if result.size != work_size:
            result = result.resize(work_size, Image.BICUBIC)
        dest = single_out if single_out is not None else out_dir / f"{src.stem}.png"
        result.save(dest)
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.reset_peak_memory_stats()
        tile_note = f" tile={args.tile}/{args.overlap}" if args.tile else ""
        print(f"OK {src.name} -> {dest} size={result.size}{tile_note}", flush=True)

    wall = time.time() - t0
    print(
        f"DONE look={args.look} n={len(images)} wall_s={wall:.2f} "
        f"device={device} peak_gb={peak:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
