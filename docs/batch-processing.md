# Larger batches and recovery

## Start a run

On Windows, start with the one-liner. Default look is **`auto`** (underwater →
spectroformer on GPU; topside → natural). It writes to a sibling `{input}-auto`
folder when `-Output` is omitted, and always resumes verified PNGs:

```powershell
.\scripts\process.ps1 "D:\path\to\NEFs"
.\scripts\process.ps1 "D:\path\to\NEFs" -Output "D:\path\to\throwaway-output"
.\scripts\process.ps1 "D:\path\to\NEFs" -Look spectroformer
.\scripts\process.ps1 "D:\path\to\NEFs" -Look vivid
```

Drop a NEF folder onto `process.cmd` for the same defaults. Classical CPU:
`-Look vivid`. UW-only GPU folder: `-Look spectroformer`. On success the
wrapper opens the output folder (`-NoOpen` to skip).

macOS:

```bash
bash scripts/process.sh /path/to/NEFs
bash scripts/process.sh /path/to/NEFs --look spectroformer
bash scripts/process.sh /path/to/NEFs --output /path/to/throwaway --no-open
```

Select a few representative photos first: a close subject, bright whites,
and a wide or hazy scene. Review full-size crops for lost detail and artificial
color before running a new collection through the same preset.

```bash
.venv/bin/python scripts/process_batch.py \
  --input '/photos/to-process' --output '/photos/results-auto' --look auto
```

If `--output` is omitted, the runner writes to a sibling `{input}-{look}` folder.
The directory scan is non-recursive and accepts `.nef` case-insensitively.
Duplicate filename stems are rejected to prevent output collisions. Original
NEFs are read-only. Processing is sequential. Neural looks use native CUDA/MPS
(or WSL fallback on Windows); classical looks stay in the local venv. Progress
lines include elapsed time and an ETA after the first processed image. The
end-of-run summary reports completed/processed/resumed/failed counts, output
size, and elapsed time.

Allow roughly 60 MB per 45 MP PNG as an initial estimate, plus room for previews,
reports, and in-progress files. Actual size depends on texture and noise.
Keep original NEFs separately; preserve earlier image versions when comparing
treatments. Ensure enough available RAM for several full-size arrays.

Current input support is Nikon NEFs with an unrotated, full-resolution embedded
JPEG matching the RAW crop. This is not a sensor-RAW developer.

## Resume verified work

```bash
.venv/bin/python scripts/process_batch.py \
  --input '/photos/to-process' --output '/photos/results-vivid' --look vivid --resume
```

The runner compares the original NEF checksum, code/dependency recipe identity,
and final PNG checksum with the completed report. It skips only verified matches
with an existing preview. Changing source bytes, code, dependencies, or preset
requires a fresh output folder. It does not silently replace incompatible files.

The summary is written after each attempted image. Image failures are recorded,
remaining inputs are attempted, and the process exits nonzero if any failed.
A failed image may leave an incomplete temporary file or an unverified output.
Use a fresh output folder for conflicting files after inspecting the failure;
automatic deletion of uncertain output is deliberately avoided.

## Locks and interrupted processes

`.processing.lock` prevents concurrent writers in one output directory and
records the local PID. Normal completion, handled errors, and Ctrl-C release
the lock. If the process is forcibly killed or the machine shuts down, the lock
can remain. If the recorded PID is not running, the error says the lock is
likely stale. Confirm no run is using that directory before manually removing
the stale lock directory. The runner does not auto-delete locks. Do not clear a
lock simply because a run is slow.

## Review and delivery

Use the JPEG comparisons to triage the batch, then inspect uncertain files at
full resolution. PNGs are the delivery files. Reports contain source filenames,
capture dates, and checksums; they are local working data and ignored by Git.
Inspect photo metadata before sharing externally.

As of September 2026, Google Photos supports PNG. Use Original quality to retain
the exported image quality. Check current limits in [Google Photos help](https://support.google.com/photos/answer/6193313)
and [backup quality guidance](https://support.google.com/photos/answer/6220791)
before a large upload. This tool does not upload photos or change account settings.
