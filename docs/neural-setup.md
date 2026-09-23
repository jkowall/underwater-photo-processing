# Neural GPU looks (Spectroformer / NU2Net)

Classical looks (`vivid`, `pop`, `natural`) need only the local `.venv`.
Default CLI look is **`auto`** (underwater → spectroformer, topside → natural).

Neural looks prefer **native GPU torch** in `.venv`:

| Platform | Install | Backend label |
| --- | --- | --- |
| Windows + NVIDIA | `requirements-neural.txt` (cu128) | `windows-cuda` |
| Apple Silicon macOS | `requirements-neural-macos.txt` | `native-mps` |
| Windows without CUDA | WSL2 `uw_eval` fallback | `wsl-uw_eval` |

## One-time machine setup

### Windows (CUDA)

1. NVIDIA driver with CUDA support (RTX 40/50-class tested).
2. From the repo root:

```powershell
.\scripts\setup.ps1                 # classical + requirements-neural.txt + fetch weights
.\scripts\fetch_neural_weights.ps1  # also called from setup; safe to re-run
```

### Apple Silicon (MPS)

```bash
bash scripts/setup.sh                 # classical + requirements-neural-macos.txt + fetch
bash scripts/fetch_neural_weights.sh  # also called from setup; safe to re-run
```

Confirm MPS:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Expected paths after a successful fetch:

- `eval/repos/spectroformer/checkpoints/best.pth`
- `eval/repos/uie_benchmark/checkpoints/UIEB/NU2Net.ckpt` (optional)

If weights are still missing, place them manually (upstream does not always
vendor checkpoints in git — see that repo’s README / releases).

## Process a folder

Windows:

```powershell
.\scripts\process.ps1 "D:\path\to\NEFs"
.\scripts\process.ps1 "D:\path\to\NEFs" -Look spectroformer  # UW-only GPU
.\scripts\process.ps1 "D:\path\to\NEFs" -Look vivid          # classical CPU
.\scripts\process.ps1 "D:\path\to\NEFs" -NoOpen              # skip Explorer
```

macOS / Linux:

```bash
bash scripts/process.sh /path/to/NEFs
bash scripts/process.sh /path/to/NEFs --look spectroformer
bash scripts/process.sh /path/to/NEFs --look vivid --no-open
```

The batch runner prints `Neural backend: windows-cuda`, `native-mps`, or
`wsl-uw_eval`. Spectroformer keeps the working aspect ratio (no 512² squash)
and reinjects source luminance detail after upsample so full-res NEFs stay sharp.

## Licensing

See [THIRD_PARTY.md](../THIRD_PARTY.md). Do not commit Spectroformer / NU2Net
weights into this Apache repo. Spectroformer’s upstream currently publishes no
LICENSE file — keep clones local; credit the WACV 2024 paper.

## Domain adaptation

When a new trip systematically needs better weights, see
[retraining.md](retraining.md). Agent checklist:
[`.cursor/skills/uie-retrain/SKILL.md`](../.cursor/skills/uie-retrain/SKILL.md).
Not required per batch.
