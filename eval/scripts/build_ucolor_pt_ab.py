#!/usr/bin/env python3
"""A/B strips: original | ucolor_tf1 | ucolor_pt | spectroformer | nu2net."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EVAL = Path("/mnt/c/Users/jkowa/underwater-photo-processing/eval")
STILLS = EVAL / "stills"
OUT = EVAL / "out"
COMPARE = EVAL / "compare" / "ucolor_pt_ab"
AS_MEDIA = Path(
    "/mnt/c/Users/jkowa/AppData/Local/Cursor/AgentStores/"
    "cursor_agent_stores/bc-120def17-2bae-43bf-b1d4-79df27017484/files/media/eval/ucolor_pt_ab"
)

METHODS = [
    ("original", None),
    ("ucolor_tf1", OUT / "ucolor"),
    ("ucolor_pt", OUT / "ucolor_pt"),
    ("spectroformer", OUT / "spectroformer"),
    ("nu2net", OUT / "nu2net"),
]


def load_rgb(path: Path, size=None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.BICUBIC)
    return im


def thumb(im: Image.Image, max_w=360) -> Image.Image:
    w, h = im.size
    if w <= max_w:
        return im
    return im.resize((max_w, int(h * max_w / w)), Image.BICUBIC)


def make_strip(stem: str) -> Image.Image | None:
    orig = load_rgb(STILLS / f"{stem}.jpg")
    size = orig.size
    panels, labels = [], []
    for name, folder in METHODS:
        if name == "original":
            panels.append(thumb(orig))
            labels.append(name)
            continue
        path = folder / f"{stem}.png"
        if not path.exists():
            print("missing", path)
            continue
        panels.append(thumb(load_rgb(path, size=size)))
        labels.append(name)
    if len(panels) < 2:
        return None
    pw, ph = panels[0].size
    label_h = 28
    canvas = Image.new("RGB", (pw * len(panels), ph + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel, (i * pw, label_h))
        draw.text((i * pw + 8, 6), label, fill=(230, 230, 230))
    return canvas


def main():
    COMPARE.mkdir(parents=True, exist_ok=True)
    AS_MEDIA.mkdir(parents=True, exist_ok=True)
    for still in sorted(STILLS.glob("*.jpg")):
        strip = make_strip(still.stem)
        if strip is None:
            continue
        dest = COMPARE / f"{still.stem}_ab.jpg"
        strip.save(dest, quality=92)
        strip.save(AS_MEDIA / f"{still.stem}_ab.jpg", quality=92)
        print("strip", dest)


if __name__ == "__main__":
    main()
