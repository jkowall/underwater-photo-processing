# Neural GPU looks (Spectroformer / NU2Net)

Classical looks (`vivid`, `pop`, `natural`) need only the Windows `.venv`.
Default **spectroformer** (and optional **nu2net**) need WSL2 + CUDA torch.

## One-time machine setup

1. Install [WSL2](https://learn.microsoft.com/windows/wsl/install) and a NVIDIA
   CUDA-capable driver that supports WSL.
2. In WSL, install [micromamba](https://mamba.readthedocs.io/) and create env
   `uw_eval` with Python 3.11 + PyTorch matching your GPU (this machine used
   torch 2.11+cu128 on an RTX 5080).
3. From the repo on the Windows filesystem (accessible as `/mnt/c/...`):

```bash
# Spectroformer (required for default look)
git clone https://github.com/Mdraqibkhan/Spectroformer.git \
  /mnt/c/Users/jkowa/underwater-photo-processing/eval/repos/spectroformer
# Place author checkpoint at:
#   eval/repos/spectroformer/checkpoints/best.pth
# (upstream ships checkpoints under checkpoints/<dataset>/ — copy or symlink
#  the UIEB/best weights you validated to checkpoints/best.pth)

# NU2Net (optional -Look nu2net)
git clone https://github.com/ddz16/Underwater-Image-Enhancement-Benchmark.git \
  /mnt/c/Users/jkowa/underwater-photo-processing/eval/repos/uie_benchmark
# Place NU2Net.ckpt at:
#   eval/repos/uie_benchmark/checkpoints/UIEB/NU2Net.ckpt
```

4. On Windows: `.\scripts\setup.ps1` — installs classical deps and prints a
   neural checklist (WSL, `uw_eval`, weight paths).

## Process a folder

```powershell
.\scripts\process.ps1 "D:\path\to\NEFs"
# sibling output: ...\NEFs-spectroformer\

.\scripts\process.ps1 "D:\path\to\NEFs" -Look auto     # UW→spectroformer, topside→natural
.\scripts\process.ps1 "D:\path\to\NEFs" -Look vivid    # classical CPU only
.\scripts\process.ps1 "D:\path\to\NEFs" -Look nu2net   # alternate GPU
```

## Licensing

See [THIRD_PARTY.md](../THIRD_PARTY.md). Do not commit Spectroformer / NU2Net
weights into this Apache repo. Spectroformer’s upstream currently publishes no
LICENSE file — keep clones local; credit the WACV 2024 paper.

## Domain adaptation

When a new trip systematically needs better weights, use the project skill
[`.cursor/skills/uie-retrain/SKILL.md`](../.cursor/skills/uie-retrain/SKILL.md)
(paired data + fine-tune + A/B gate). Not required per batch.
