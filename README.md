# Underwater Photo Processing

A Python processor and Codex skill for correcting underwater photographs:
reduce blue/green cast, restore warmer color, and export full-resolution
lossless PNGs.

The default **spectroformer** look runs a GPU underwater-image-enhancement
model (bakeoff runner-up; best shippable GPU path after the 2026 UIE eval).
It needs **WSL2 + micromamba env `uw_eval`** with CUDA torch. Classical CPU
looks (`vivid`, `pop`, `natural`) still run in the Windows `.venv` with
OpenCV/NumPy only — no API keys or cloud services.

**Current input support:** Nikon NEFs containing an unrotated, full-resolution
embedded JPEG whose dimensions match the RAW crop. This is deliberately the
tested camera-rendering workflow, not a sensor-RAW developer. PNG avoids further
lossy compression but cannot recover information missing from that JPEG.

## Quick start

On Windows, one command processes a NEF folder with **spectroformer** (GPU via WSL).
Output goes to a sibling `{folder}-spectroformer` directory so originals stay read-only.
The wrapper creates a Python 3.12 virtual environment if needed and always
passes `--resume` so reruns skip verified PNGs.

```powershell
git clone https://github.com/jkowall/underwater-photo-processing.git
cd underwater-photo-processing
.\scripts\setup.ps1
.\scripts\process.ps1 "D:\path\to\NEFs"
```

`setup.ps1` installs classical deps and prints a checklist for WSL / `uw_eval` /
Spectroformer + NU2Net weights under `eval/repos/`. Details:
[docs/neural-setup.md](docs/neural-setup.md).

Drop a NEF folder onto `process.cmd` for the same default. Pass `-Output` for a
throwaway destination, or `-Look auto` for mixed dive-day folders (underwater
→ spectroformer; topside/sunset → natural). Classical CPU: `-Look vivid`.
Do not point `-Output` at an existing corrected batch unless you intend to resume it.

macOS/Linux classical path (neural looks still expect WSL/`uw_eval` on this machine):

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
Originals are read-only, and existing results are never silently overwritten.

## Presets

| Look | Treatment |
| --- | --- |
| `spectroformer` (default) | GPU UIE @ long-edge 1536 (infer 512→upsample); WSL `uw_eval` |
| `nu2net` | Fast GPU alternate; pad-to-16; WSL `uw_eval` |
| `natural` | Classical: water-cast correction, red compensation, WB, local contrast |
| `pop` | Classical: natural + mild denoise, particle cleanup, stronger contrast |
| `vivid` | Classical: pop + richer color (approved OpenCV recipe) |
| `auto` | Underwater → `spectroformer`; topside/sunset → `natural` |

Neural looks enhance at long-edge 1536 then upsample to the source embedded-JPEG
size for full-res PNG export. Classical looks process at full embedded resolution.

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
and known limitations. Mixed dive-day folders can use `--look auto`.

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

Read [the pipeline](docs/pipeline.md) for classical correction stages and
[the current treatment](references/current-treatment.md) for the OpenCV recipe.
Neural looks use `eval/scripts/run_uie_look.py` (Spectroformer / NU2Net).
Inspect bright whites, fine subject detail, and open-water areas in previews
and full-resolution crops.

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

**Credits / third parties:** see [NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md).
Spectroformer (default GPU look) and NU2Net keep their upstream copyright;
Spectroformer’s GitHub repo currently publishes **no LICENSE file** — do not
bundle those weights into an Apache redistribution without author permission.
Classical `vivid` / `pop` / `natural` looks are this project’s Apache code only.

Other dependencies (NumPy, OpenCV, rawpy, PyTorch, etc.) retain their own licenses.
