# Neural GPU looks (Spectroformer / NU2Net)

Classical looks (`vivid`, `pop`, `natural`) need only the Windows `.venv`.
Default CLI look is **`auto`** (underwater → spectroformer, topside → natural).
Spectroformer / NU2Net need WSL2 + CUDA torch.

## One-time machine setup

1. Install [WSL2](https://learn.microsoft.com/windows/wsl/install) and a NVIDIA
   CUDA-capable driver that supports WSL.
2. In WSL, install [micromamba](https://mamba.readthedocs.io/) and create env
   `uw_eval` with Python 3.11 + PyTorch matching your GPU (this machine used
   torch 2.11+cu128 on an RTX 5080).
3. On Windows, from the repo root:

```powershell
.\scripts\setup.ps1                 # classical venv + calls fetch below
.\scripts\fetch_neural_weights.ps1  # clone repos; promote/copy checkpoints
```

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

## Licensing

See [THIRD_PARTY.md](../THIRD_PARTY.md). Do not commit Spectroformer / NU2Net
weights into this Apache repo. Spectroformer’s upstream currently publishes no
LICENSE file — keep clones local; credit the WACV 2024 paper.

## Domain adaptation

When a new trip systematically needs better weights, see
[retraining.md](retraining.md). Agent checklist:
[`.cursor/skills/uie-retrain/SKILL.md`](../.cursor/skills/uie-retrain/SKILL.md).
Not required per batch.
