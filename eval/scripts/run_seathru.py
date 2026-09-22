"""Classic SeaThru-inspired recovery using RGB + metric depth."""
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_depth_m(path_png: Path, path_npy: Path | None = None) -> np.ndarray:
    if path_npy is not None and path_npy.exists():
        return np.load(path_npy).astype(np.float32)
    d = cv2.imread(str(path_png), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"missing depth {path_png}")
    if d.dtype == np.uint16:
        return d.astype(np.float32) / 1000.0
    return d.astype(np.float32)


def estimate_B(img: np.ndarray, depth: np.ndarray, far_pct: float = 85) -> np.ndarray:
    thr = np.percentile(depth, far_pct)
    mask = depth >= thr
    if mask.sum() < 50:
        mask = depth >= np.percentile(depth, 70)
    B = np.zeros(3, np.float32)
    for c in range(3):
        B[c] = np.median(img[:, :, c][mask])
    return np.clip(B, 1e-3, 1.0)


def estimate_beta(
    img: np.ndarray, depth: np.ndarray, B: np.ndarray, near_pct: float = 20, far_pct: float = 80
) -> np.ndarray:
    z_near = np.percentile(depth, near_pct)
    z_far = np.percentile(depth, far_pct)
    beta = np.zeros(3, np.float32)
    for c in range(3):
        near = img[:, :, c][(depth <= z_near) & (depth > 0)]
        far = img[:, :, c][(depth >= z_far)]
        if len(near) < 20 or len(far) < 20:
            beta[c] = 0.1
            continue
        dn = np.clip(np.median(near) - B[c], 1e-4, 1)
        df = np.clip(np.median(far) - B[c], 1e-4, 1)
        dz = max(float(z_far - z_near), 0.5)
        ratio = float(np.clip(df / dn, 1e-3, 0.999))
        beta[c] = float(np.clip(-np.log(ratio) / dz, 0.02, 1.5))
    return beta


def seathru_recover(img: np.ndarray, depth: np.ndarray):
    img = img.astype(np.float32)
    depth = np.maximum(depth.astype(np.float32), 0.1)
    B = estimate_B(img, depth)
    beta = estimate_beta(img, depth, B)
    gamma = beta * 0.9
    J = np.zeros_like(img)
    for c in range(3):
        atten = np.exp(-beta[c] * depth)
        back = B[c] * (1.0 - np.exp(-gamma[c] * depth))
        direct = img[:, :, c] - back
        J[:, :, c] = direct / np.maximum(atten, 1e-3)
    J = np.clip(J, 0, None)
    for c in range(3):
        lo, hi = np.percentile(J[:, :, c], [1, 99])
        if hi <= lo:
            hi = lo + 1e-3
        J[:, :, c] = np.clip((J[:, :, c] - lo) / (hi - lo), 0, 1)
    return J, B, beta


def main():
    stills = Path("/mnt/c/Users/jkowa/underwater-photo-processing/eval/stills")
    depth_dir = Path("/mnt/c/Users/jkowa/underwater-photo-processing/eval/out/deepsee/depth")
    depth_vis = Path("/mnt/c/Users/jkowa/underwater-photo-processing/eval/out/deepsee/depth_vis")
    out = Path("/mnt/c/Users/jkowa/underwater-photo-processing/eval/out/seathru")
    log_path = Path("/mnt/c/Users/jkowa/underwater-photo-processing/eval/logs/uie_bakeoff/seathru.log")
    out.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ok = 0
    lines = []
    for p in sorted(stills.glob("*.jpg")):
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        npy = depth_vis / f"{p.stem}_raw_depth_meter.npy"
        png = depth_dir / f"{p.stem}.png"
        depth = load_depth_m(png, npy if npy.exists() else None)
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        J, B, beta = seathru_recover(rgb, depth)
        Image.fromarray((J * 255).astype(np.uint8)).save(out / f"{p.stem}.png")
        ok += 1
        lines.append(f"{p.stem} B={B.tolist()} beta={beta.tolist()} depth_mean={float(depth.mean()):.2f}")
        print("OK", p.name, "B", B, "beta", beta)

    wall = time.time() - t0
    log_path.write_text(
        f"STATUS=SUCCESS\nwall_s={wall:.1f}\nn={ok}\n"
        f"note=classic SeaThru-inspired solver on DA-V2 metric depths\n" + "\n".join(lines) + "\n"
    )
    print("seathru done", ok, wall)


if __name__ == "__main__":
    main()
