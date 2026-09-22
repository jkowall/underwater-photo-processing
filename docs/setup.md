# Setup

## Environment

Use Python 3.12 and a dedicated virtual environment. The pinned versions in
`requirements.txt` reproduce the reference dependency set. Do not install into
system Python or upload photos to a service to use this processor.

On macOS (Apple Silicon):

```bash
bash scripts/setup.sh
bash scripts/process.sh /path/to/NEFs
```

`setup.sh` installs classical deps, MPS torch (`requirements-neural-macos.txt`),
and fetches neural repos when possible. Full neural setup:
[neural-setup.md](neural-setup.md).

On macOS/Linux classical-only:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/process_batch.py --help
```

`bash scripts/setup.sh` performs the same classical steps (and neural on Darwin).
Set `PYTHON_BIN` if the interpreter has another name.

`.\scripts\setup.ps1` installs classical deps into `.venv` and prints a
checklist for neural GPU looks (Windows CUDA, WSL fallback, Spectroformer
and NU2Net checkpoints under `eval/repos/`). Full neural setup:
[neural-setup.md](neural-setup.md). Then:

```powershell
.\scripts\process.ps1 "D:\path\to\NEFs"
```

Default look is **auto** (sibling `{input}-auto`). UW-only GPU:
`-Look spectroformer`. Classical CPU: `-Look vivid`. Fast GPU alternate:
`-Look nu2net`. The wrapper always passes `--resume` and opens the output
folder on success (`-NoOpen` to skip). Drop a folder onto `process.cmd`
for the same defaults.

Windows neural looks prefer CUDA torch (`requirements-neural.txt`); WSL
`uw_eval` is a fallback. Apple Silicon uses MPS
(`requirements-neural-macos.txt`). Binary-wheel availability for classical
deps can depend on your platform and Python version.

## Optional agent skill

Run `scripts/install_skill.py` using Python. It copies `SKILL.md`, optional
`agents/` metadata, treatment references, and the two processor scripts. It
does not copy your virtual environment, photographs, Git metadata, or generated
outputs. The skill works with any agent that reads `SKILL.md` (Cursor in-repo
by default; Codex or others via `--destination`).

An existing install is not overwritten unless you pass `--update`. Updates
replace known payload files and preserve unrelated files. Review custom changes
first. Examples:

```bash
.venv/bin/python scripts/install_skill.py
.venv/bin/python scripts/install_skill.py --destination ~/.cursor/skills/underwater-photo-processing --update
```

## Troubleshooting

- **Missing module:** use the Python executable inside the configured virtual
  environment, including when an agent invokes the installed skill.
- **Spectroformer / nu2net fails:** ensure GPU torch is installed
  (`requirements-neural.txt` → `torch.cuda.is_available()`, or on Mac
  `requirements-neural-macos.txt` → `torch.backends.mps.is_available()`),
  weights exist at `eval/repos/spectroformer/checkpoints/best.pth` (and NU2Net
  path if used), and/or WSL `uw_eval` works as Windows fallback. Re-run
  `.\scripts\setup.ps1` or `bash scripts/setup.sh`. Fall back with `-Look vivid`.
- **Unsupported NEF or missing full-size JPEG:** this release does not implement
  a sensor-RAW decoder fallback. Use a compatible source or develop the RAW in
  a RAW-capable application; JPEG/TIFF imports are not supported by this CLI yet.
- **Rotated orientation:** the tested recipe rejects rotated RAW orientation.
  It will not silently rotate, crop, or upscale the source to pass a size check.
- **Insufficient memory:** process sequentially and close memory-heavy apps.
  Preview creation does not lower the full-resolution processing requirement.
