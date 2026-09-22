# Underwater Photo Processing

A Python processor and agent skill for correcting underwater photographs:
reduce blue/green cast, restore warmer color, and export full-resolution
lossless PNGs.

The default **`auto`** look routes underwater frames through **spectroformer**
(GPU via WSL) and topside/sunset through classical **natural**, so mixed dive
days are safer. Force `-Look spectroformer` for UW-only folders. Classical CPU looks
(`vivid`, `pop`, `natural`) run in the Windows `.venv` with OpenCV/NumPy only —
no API keys or cloud services. Spectroformer prefers **Windows CUDA torch** in
the same `.venv` (`requirements-neural.txt`); **WSL2 `uw_eval`** remains a
fallback if native CUDA is unavailable.

**Current input support:** Nikon NEFs containing an unrotated, full-resolution
embedded JPEG whose dimensions match the RAW crop. This is deliberately the
tested camera-rendering workflow, not a sensor-RAW developer. PNG avoids further
lossy compression but cannot recover information missing from that JPEG.

## Quick start

On Windows, one command processes a NEF folder with **`auto`** (sibling
`{folder}-auto`). The wrapper creates a Python 3.12 virtual environment if
needed and always passes `--resume` so reruns skip verified PNGs.

```powershell
git clone https://github.com/jkowall/underwater-photo-processing.git
cd underwater-photo-processing
.\scripts\setup.ps1
.\scripts\fetch_neural_weights.ps1   # also invoked from setup.ps1
.\scripts\process.ps1 "D:\path\to\NEFs"
```

`setup.ps1` installs classical deps, clones neural repos when possible, and
prints a WSL / `uw_eval` / weight checklist. Details:
[docs/neural-setup.md](docs/neural-setup.md).

Drop a NEF folder onto `process.cmd` for the same default. Pass `-Output` for a
throwaway destination. UW-only: `-Look spectroformer`. Classical CPU: `-Look vivid`.
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
| `auto` (default) | Underwater → `spectroformer`; topside/sunset → `natural` |
| `spectroformer` | GPU UIE @ long-edge 1536 (infer 512→upsample); WSL `uw_eval` |
| `nu2net` | Fast GPU alternate; pad-to-16; WSL `uw_eval` |
| `natural` | Classical: water-cast correction, red compensation, WB, local contrast |
| `pop` | Classical: natural + mild denoise, particle cleanup, stronger contrast |
| `vivid` | Classical: pop + richer color (approved OpenCV recipe) |

Neural looks enhance at long-edge 1536 then upsample to the source embedded-JPEG
size for full-res PNG export. Classical looks process at full embedded resolution.

**Retraining:** not part of normal processing. See
[docs/retraining.md](docs/retraining.md) (occasional domain adaptation only).

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

## Agent skill (Cursor, Codex, and other SKILL.md hosts)

The repo-root [`SKILL.md`](SKILL.md) is a standard Agent Skill entrypoint — not
tied to one LLM product. Cursor picks it up from the repo (and
[`.cursor/skills/uie-retrain/`](.cursor/skills/uie-retrain/) for domain
adaptation). Other hosts can copy the same files.

```bash
# Optional: install a personal copy (Codex default path, or pass --destination)
.venv/bin/python scripts/install_skill.py
.venv/bin/python scripts/install_skill.py --destination ~/.cursor/skills/underwater-photo-processing
```

`install_skill.py` copies `SKILL.md`, `agents/`, `references/`, and the two
processor scripts. Use `--update` after reviewing local edits. Optional Codex
metadata lives in [`agents/openai.yaml`](agents/openai.yaml); ignore it on
hosts that do not use that file.
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
