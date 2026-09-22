---
name: underwater-photo-processing
description: Correct underwater photos and batches in Python, including water cast, attenuated reds, particulate cleanup, denoising, and lossless PNG export. Use when processing dive photographs or refining their color treatment.
---

# Underwater photo processing

Process the actual image pixels when the user requests this computational
workflow. Classical looks use OpenCV/NumPy locally. GPU looks
(`spectroformer`, `nu2net`) call WSL `uw_eval` models — credit upstream authors
per [THIRD_PARTY.md](THIRD_PARTY.md).

## Preferences and current state

- Deliver full-resolution lossless PNG with an sRGB profile and original camera
  capture dates. Keep originals and earlier accepted/reviewable versions.
- Default CLI look is **`auto`** (spectroformer underwater, natural topside).
  Classical maintainer-approved OpenCV recipe remains `--look vivid`.
  Force `--look spectroformer` for UW-only folders. Neural looks prefer
  Windows CUDA in `.venv`; WSL `uw_eval` is fallback.
- You do **not** retrain models per photo batch; run inference with existing
  weights. Retrain only for deliberate domain adaptation with paired data.
  When the user asks to retrain / fine-tune / domain-adapt UIE weights, follow
  the project skill [`.cursor/skills/uie-retrain/SKILL.md`](.cursor/skills/uie-retrain/SKILL.md).
- Read [references/current-treatment.md](references/current-treatment.md) for
  the classical recipe. Assess representative images for a new batch rather
  than assuming every scene needs identical correction.
- When authorized to maintain this skill, incorporate material feedback into
  the treatment reference and any changed helper. Keep settled decisions concise;
  replace superseded recommendations. Do not infer permission for unrelated edits.
## Working method

Inspect source format, dimensions, color profile, channel statistics, and several
representative scenes before choosing a treatment. Re-estimate ambient light
and white balance per photo. A single close-up's settings do not transfer
unchanged to wide reef views or different lighting.

Prefer actual RAW decoding when available. The tested Nikon Z8 NEFs use a
compression unsupported by the installed LibRaw and macOS decoder. Their
embedded camera JPEGs match the full RAW crop dimensions. The bundled helper
supports that fallback explicitly and rejects smaller previews. Do not silently
upscale or imply that an embedded JPEG is sensor RAW. Inverse-sRGB is linearized
camera rendering, not original sensor-linear data. PNG avoids additional lossy
compression but cannot restore lost source information.

Use conservative ambient-veiling-light subtraction, adaptive red compensation,
linear-RGB Shades of Gray white balance, and LAB luminance CLAHE. Missing red
values and backscatter cannot be uniquely recovered from one image; label the
result as an estimate. Protect clipped cyan highlights from artificial pink
patches, and check saturated colors after adjusting contrast or vibrance.

Separate noise from visible particulate. Mild edge-preserving luminance and
chroma denoising reduces noise. Compact bright outliers in smooth backgrounds
can be locally interpolated; broad backscatter and particles next to subject
details need more selective work. Avoid smoothing coral/anemone tips, fish
patterns, and fine appendages. Report automated detections as candidates,
not confirmed particles. Review full-size crops as well as contact sheets.

## Reusable processor

The self-contained Python helpers are in `scripts/`. Dependencies are NumPy,
OpenCV, rawpy, and Pillow. Use an existing compatible environment or create a
task-local virtual environment; do not install packages into system Python.

```sh
python scripts/process_batch.py --input '/path/to/NEFs' --output '/path/to/new-output' --look vivid
```

On Windows, prefer the wrapper (always `--look vivid` unless overridden, sibling
`{input}-vivid` output if `-Output` is omitted, always `--resume`):

```powershell
.\scripts\process.ps1 "D:\path\to\NEFs"
.\scripts\process.ps1 "D:\path\to\NEFs" -Look auto -Output "D:\path\to\throwaway"
```

Drop a NEF folder onto `process.cmd` for the same defaults.

`--input` accepts one NEF or a directory of NEFs. If `--output` is omitted,
results go to a sibling `{input}-{look}` folder. `--look natural` applies the
initial correction; `pop` adds second-pass cleanup and contrast; the default
`vivid` adds the approved substantial chroma increase. `--look auto` keeps
underwater frames on `vivid` and uses `natural` for topside/sunset so mixed
dive-day folders are not over-boosted. These are starting recipes, not a claim
of calibrated color accuracy. Use a small representative selection for an
unfamiliar batch. The helper refuses unverified output collisions; `--resume`
skips only files matching source, recipe, dependencies and output checksums.
Progress includes elapsed time and an ETA; the end-of-run summary reports
counts, errors, output size, and elapsed time. A stale lock whose recorded PID
is not running is reported as likely stale; do not auto-delete locks.
It writes PNGs, before/
after previews, reports, and a summary. It does not upload or publish anything.

Use the named executable belonging to the environment, such as `.venv/bin/python`.

## Verification and delivery

Verify the complete input/output count, original pixel dimensions, PNG integrity,
exact decoded pixel equality, sRGB profile, and capture date. Record channel
means, standard deviations/histograms, and clipping before/after when diagnosing
correction. Visual review must include a close subject, bright whites, and an
open-water or hazy scene when present. Use equivalent framing and zoom for
comparisons; distinguish preview-resolution tuning from full-resolution output.

Link the delivery folder and useful comparisons. Name residual limitations
briefly. Preserve the source originals regardless of aesthetic approval.
Google Photos PNG support and Original quality were verified in September 2026;
recheck current official requirements when giving new compatibility advice.
