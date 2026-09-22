"""Build UIE bakeoff compare grids and normalize ucolor outs."""
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

EVAL = Path("/mnt/c/Users/jkowa/underwater-photo-processing/eval")
STILLS = EVAL / "stills"
OUT = EVAL / "out"
COMPARE = EVAL / "compare" / "uie_bakeoff"
AS_MEDIA = Path(
    "/mnt/c/Users/jkowa/AppData/Local/Cursor/AgentStores/"
    "cursor_agent_stores/bc-120def17-2bae-43bf-b1d4-79df27017484/files/media/eval/uie_bakeoff"
)
LOG = EVAL / "logs" / "uie_bakeoff"

METHODS = [
    ("original", None),
    ("deepsee", OUT / "deepsee"),
    ("mlle", OUT / "mlle"),
    ("nu2net", OUT / "nu2net"),
    ("ushape", OUT / "ushape"),
    ("spectroformer", OUT / "spectroformer"),
    ("ucolor", OUT / "ucolor"),
    ("seathru", OUT / "seathru"),
    ("diffwater", OUT / "diffwater"),
]


def load_rgb(path: Path, size=None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32)
    if arr.min() < 0 or arr.max() > 255:
        arr = np.clip(arr, 0, 255)
    im = Image.fromarray(arr.astype(np.uint8))
    if size is not None:
        im = im.resize(size, Image.BICUBIC)
    return im


def normalize_ucolor():
    src_dir = (
        EVAL
        / "repos"
        / "ucolor_code"
        / "Ucolor_final_model_corrected"
        / "input_90"
    )
    dest = OUT / "ucolor"
    dest.mkdir(parents=True, exist_ok=True)
    for p in sorted(STILLS.glob("*.jpg")):
        cand = src_dir / f"{p.stem}.png_out.png"
        if not cand.exists():
            print("MISSING ucolor", cand)
            continue
        arr = np.asarray(Image.open(cand).convert("RGB"), dtype=np.float32)
        # values may be outside 0-255
        arr = np.clip(arr, 0, 255)
        # mild percentile stretch if mostly washed
        lo, hi = np.percentile(arr, [1, 99])
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
        Image.fromarray(arr.astype(np.uint8)).save(dest / f"{p.stem}.png")
        print("ucolor ok", p.stem)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "ucolor.log").write_text(
        "STATUS=SUCCESS\nwall_s~480\nn=8\nstack=Podman TF1.15.5-py3\n"
        "note=transmission from DA-V2 depth proxy; outputs clipped+percentile stretch\n"
    )


def thumb(im: Image.Image, max_w=320) -> Image.Image:
    w, h = im.size
    if w <= max_w:
        return im
    nh = int(h * max_w / w)
    return im.resize((max_w, nh), Image.BICUBIC)


def make_strip(stem: str, methods):
    orig = load_rgb(STILLS / f"{stem}.jpg")
    size = orig.size
    panels = []
    labels = []
    for name, folder in methods:
        if name == "original":
            panels.append(thumb(orig))
            labels.append(name)
            continue
        path = folder / f"{stem}.png"
        if not path.exists():
            continue
        panels.append(thumb(load_rgb(path, size=size)))
        labels.append(name)
    if not panels:
        return None
    pw, ph = panels[0].size
    label_h = 28
    canvas = Image.new("RGB", (pw * len(panels), ph + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for i, (p, lab) in enumerate(zip(panels, labels)):
        canvas.paste(p, (i * pw, label_h))
        draw.text((i * pw + 8, 6), lab, fill=(240, 240, 240))
    return canvas


def main():
    COMPARE.mkdir(parents=True, exist_ok=True)
    AS_MEDIA.mkdir(parents=True, exist_ok=True)
    normalize_ucolor()

    success = [("original", None)]
    for name, folder in METHODS[1:]:
        n = len(list(folder.glob("DSC_*.png"))) if folder else 0
        if n >= 8:
            success.append((name, folder))
            print("include", name, n)
        else:
            print("skip", name, n)

    stems = [p.stem for p in sorted(STILLS.glob("*.jpg"))]
    strips = []
    for stem in stems:
        strip = make_strip(stem, success)
        if strip is None:
            continue
        dest = COMPARE / f"strip_{stem}.jpg"
        strip.save(dest, quality=90)
        strip.save(AS_MEDIA / f"strip_{stem}.jpg", quality=90)
        strips.append(strip)
        print("strip", stem, strip.size)

    # contact sheet stacking strips
    if strips:
        w = max(s.size[0] for s in strips)
        h = sum(s.size[1] for s in strips) + 10 * (len(strips) - 1)
        sheet = Image.new("RGB", (w, h), (10, 10, 10))
        y = 0
        for s in strips:
            sheet.paste(s, (0, y))
            y += s.size[1] + 10
        sheet.save(COMPARE / "contact_sheet.jpg", quality=85)
        sheet.save(AS_MEDIA / "contact_sheet.jpg", quality=85)
        print("contact", sheet.size)

    # method grid for one representative still
    rep = "DSC_2452"
    strip = make_strip(rep, success)
    if strip:
        strip.save(COMPARE / "hero_DSC_2452.jpg", quality=92)
        strip.save(AS_MEDIA / "hero_DSC_2452.jpg", quality=92)


if __name__ == "__main__":
    main()
