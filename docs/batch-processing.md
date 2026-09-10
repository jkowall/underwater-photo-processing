# Larger batches and recovery

## Start a run

Select a few representative photos first: a close subject, bright whites,
and a wide or hazy scene. Review full-size crops for lost detail and artificial
color before running a new collection through the same preset.

```bash
.venv/bin/python scripts/process_batch.py \
  --input '/photos/to-process' --output '/photos/results-vivid' --look vivid
```

The directory scan is non-recursive and accepts `.nef` case-insensitively.
Duplicate filename stems are rejected to prevent output collisions. Original
NEFs are read-only. Processing is sequential; no per-image model request occurs.

Allow roughly 60 MB per 45 MP PNG as an initial estimate, plus room for previews,
reports, and in-progress files. Actual size depends on texture and noise.
Keep original NEFs separately; preserve earlier image versions when comparing
treatments. Ensure enough available RAM for several full-size arrays.

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
can remain. Confirm no run is using that directory before manually removing
the stale lock directory. Do not clear a lock simply because a run is slow.

## Review and delivery

Use the JPEG comparisons to triage the batch, then inspect uncertain files at
full resolution. PNGs are the delivery files. Reports contain source filenames,
capture dates, and checksums; they are local working data and ignored by Git.
Inspect photo metadata before sharing externally.

As of September 2026, Google Photos supports PNG. Use Original quality to retain
the exported image quality. Check current limits in [Google Photos help](https://support.google.com/photos/answer/6193313)
and [backup quality guidance](https://support.google.com/photos/answer/6220791)
before a large upload. This tool does not upload photos or change account settings.
