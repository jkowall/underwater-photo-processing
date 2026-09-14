# Underwater Photo Processing

A Python processor and Codex skill for correcting underwater photographs:
reduce blue/green cast, restore warmer color, clean small bright particles,
and export full-resolution lossless PNGs.

The default **vivid** look combines color correction, conservative cleanup,
contrast, and a substantial color boost. It was refined on a 26-photo Nikon Z8
batch and approved by the maintainer. All image processing runs locally with
OpenCV and NumPy; it requires no AI API calls, API keys, or subscriptions.
Using Codex to orchestrate or visually review a batch still consumes model usage.

**Current input support:** Nikon NEFs containing an unrotated, full-resolution
embedded JPEG whose dimensions match the RAW crop. This is deliberately the
tested camera-rendering workflow, not a sensor-RAW developer. PNG avoids further
lossy compression but cannot recover information missing from that JPEG.

## Quick start

On Windows, one command processes a NEF folder with the approved **vivid** look.
Output goes to a sibling `{folder}-vivid` directory so originals stay read-only.
The wrapper creates a Python 3.12 virtual environment if needed and always
passes `--resume` so reruns skip verified PNGs.

```powershell
git clone https://github.com/jkowall/underwater-photo-processing.git
cd underwater-photo-processing
.\scripts\setup.ps1
.\scripts\process.ps1 "D:\path\to\NEFs"
```

Drop a NEF folder onto `process.cmd` for the same default. Pass `-Output` for a
throwaway destination, or `-Look auto` for mixed dive-day folders (underwater
stays vivid; topside/sunset uses natural). Do not point `-Output` at an existing
corrected batch unless you intend to resume it.

macOS/Linux, or calling the Python runner directly:

```bash
git clone https://github.com/jkowall/underwater-photo-processing.git
cd underwater-photo-processing
bash scripts/setup.sh

.venv/bin/python scripts/process_batch.py \
  --input '/path/to/NEFs' \
  --output '/path/to/new-results' \
  --look vivid
```

If `--output` is omitted, results go to a sibling `{input}-{look}` folder.
If Python has a different executable name, run `PYTHON_BIN=python3 bash scripts/setup.sh`.
Alternatively, create a virtual environment and install `requirements.txt`
directly. See [setup](docs/setup.md) for details.

Start with a representative selection before running a large collection.
The input can be one `.NEF` or a directory; directory scans are not recursive.
Each photo receives separately estimated correction parameters. Originals are
read-only, and existing results are never silently overwritten.

## Presets

| Look | Treatment |
| --- | --- |
| `natural` | Water-cast correction, red compensation, white balance, local contrast |
| `pop` | Natural plus mild denoising, small-particle cleanup, stronger contrast |
| `vivid` (default) | Pop plus visibly richer color, with protection for near-neutral whites |
| `auto` | Per-image guess from a small preview: underwater → `vivid`, topside/sunset → `natural` |

These are creative starting points. White balance and missing-red recovery are
single-image estimates, not calibrated measurements of the subject's true color.

## Outputs and larger batches

- Full-resolution 8-bit RGB PNGs with sRGB profiles and original capture dates.
- Individual before/after JPEG previews for review.
- Per-image reports with histograms, correction parameters, checksums, and
  metadata/pixel verification results.
- A batch summary updated after every attempted image.

Resume a compatible interrupted run with the same command plus `--resume`.
`.\scripts\process.ps1` always includes `--resume`. Completed PNGs are skipped
only if their source, processing code, dependencies, and output checksums match.
A changed source or recipe requires a fresh output folder. An output-directory
lock prevents concurrent writers. If a lock remains after a killed process and
the recorded PID is not running, the error message says the lock is likely stale;
do not delete it until you confirm no run is using that folder.

See [batch processing](docs/batch-processing.md) for recovery, disk planning,
and known limitations. Mixed dive-day folders can use `--look auto` so sunsets
are not over-boosted; default `vivid` still applies the approved recipe to every
file. In the reference batch, 26 vivid PNGs totaled about
1.62 GB. Sequential processing bounds memory; large 45 MP frames still require
several GB of available RAM. Performance varies by machine and scene.

## Install the Codex skill

```bash
.venv/bin/python scripts/install_skill.py
```

This installs the skill and its self-contained processing helpers under
`$CODEX_HOME/skills/underwater-photo-processing`, or `~/.codex/skills` when unset.
Use `--update` to refresh an existing installation after reviewing local edits.
Invoke it as `$underwater-photo-processing` in Codex. The [skill entrypoint](SKILL.md)
records the approved look and the instruction to keep its treatment notes current.

## How it works and what to inspect

Read [the pipeline](docs/pipeline.md) for the correction stages and
[the current treatment](references/current-treatment.md) for the approved recipe.
Inspect bright whites, fine subject detail, and open-water areas in previews
and full-resolution crops. Conservative cleanup can leave larger diffuse
backscatter; stronger automatic removal can erase real detail.

The repository contains code and documentation only. Photos, generated output,
and machine-specific paths are excluded. Capture metadata in your own outputs
can contain personal information; review it before sharing the images.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions should preserve the
approved preset unless a treatment change is intentional and visually reviewed.

## License

Copyright 2026 Jonah Kowall. Licensed under the [Apache License 2.0](LICENSE).
Dependencies retain their own licenses; this license does not relicense them.
