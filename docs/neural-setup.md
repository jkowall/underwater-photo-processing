# Neural GPU looks (Spectroformer / NU2Net)

Classical looks (`vivid`, `pop`, `natural`) need only the Windows `.venv`.
Default CLI look is **`auto`** (underwater → spectroformer, topside → natural).

Neural looks prefer **Windows CUDA torch** in `.venv`. WSL2 `uw_eval` is a
fallback.

## One-time machine setup

1. NVIDIA driver with CUDA support (RTX 40/50-class tested).
2. From the repo root on Windows:

```powershell
.\scripts\setup.ps1                 # classical + requirements-neural.txt + fetch weights
.\scripts\fetch_neural_weights.ps1  # also called from setup; safe to re-run
```

`setup.ps1` installs `requirements.txt` then `requirements-neural.txt`
(PyTorch cu128 wheels). If that pip step fails, install WSL2 + micromamba env
`uw_eval` as a fallback (same torch stack we used in bakeoffs).

`fetch_neural_weights.ps1` clones Spectroformer and uie_benchmark into
`eval/repos/` when missing, and copies a found `*.pth` to
`eval/repos/spectroformer/checkpoints/best.pth` when possible. If weights are
still missing, place them manually (upstream does not always vendor checkpoints
in git — see that repo’s README / releases).

Expected paths after a successful fetch:

- `eval/repos/spectroformer/checkpoints/best.pth`
- `eval/repos/uie_benchmark/checkpoints/UIEB/NU2Net.ckpt` (optional)

## Process a folder

```powershell
.\scripts\process.ps1 "D:\path\to\NEFs"
# sibling output: ...\NEFs-auto\

.\scripts\process.ps1 "D:\path\to\NEFs" -Look spectroformer  # UW-only GPU
.\scripts\process.ps1 "D:\path\to\NEFs" -Look vivid          # classical CPU
.\scripts\process.ps1 "D:\path\to\NEFs" -Look nu2net         # alternate GPU
```

The batch runner prints `Neural backend: windows-cuda` or `wsl-uw_eval`.

## Licensing

See [THIRD_PARTY.md](../THIRD_PARTY.md). Do not commit Spectroformer / NU2Net
weights into this Apache repo. Spectroformer’s upstream currently publishes no
LICENSE file — keep clones local; credit the WACV 2024 paper.

## Domain adaptation

When a new trip systematically needs better weights, see
[retraining.md](retraining.md). Agent checklist:
[`.cursor/skills/uie-retrain/SKILL.md`](../.cursor/skills/uie-retrain/SKILL.md).
Not required per batch.
