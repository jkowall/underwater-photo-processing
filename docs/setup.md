# Setup

## Environment

Use Python 3.12 and a dedicated virtual environment. The pinned versions in
`requirements.txt` reproduce the reference dependency set. Do not install into
system Python or upload photos to a service to use this processor.

On macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/process_batch.py --help
```

`bash scripts/setup.sh` performs the same steps. Set `PYTHON_BIN` if the
interpreter has another name. The shell helper does not install Python itself.

`.\scripts\setup.ps1` performs the same steps on Windows (via `py -3.12`, or
`PYTHON_BIN` if set). Then:

```powershell
.\scripts\process.ps1 "D:\path\to\NEFs"
```

If `-Output` is omitted, PNGs go to a sibling `{input}-vivid` folder. The
wrapper always passes `--resume`. Drop a folder onto `process.cmd` for the
same defaults. Windows instructions describe the standard Python setup;
full-resolution NEF processing was originally validated on macOS, not Windows.
Binary-wheel availability can depend on your platform and Python version.

## Optional Codex skill

Run `scripts/install_skill.py` using Python. It copies only `SKILL.md`, skill
metadata, treatment references, and the two processor scripts. It does not copy
your virtual environment, photographs, Git metadata, or generated outputs.
The installed skill can use any environment with the required dependencies.

An existing skill is not overwritten unless you pass `--update`. Updates replace
known payload files and preserve unrelated files. Review custom changes first.
Use `--destination /path/to/skill-folder` to test an installation elsewhere.

## Troubleshooting

- **Missing module:** use the Python executable inside the configured virtual
  environment, including when an agent invokes the installed skill.
- **Unsupported NEF or missing full-size JPEG:** this release does not implement
  a sensor-RAW decoder fallback. Use a compatible source or develop the RAW in
  a RAW-capable application; JPEG/TIFF imports are not supported by this CLI yet.
- **Rotated orientation:** the tested recipe rejects rotated RAW orientation.
  It will not silently rotate, crop, or upscale the source to pass a size check.
- **Insufficient memory:** process sequentially and close memory-heavy apps.
  Preview creation does not lower the full-resolution processing requirement.
