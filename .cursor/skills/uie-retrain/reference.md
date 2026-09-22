# UIE retrain — reference

## Key paths

| Item | Path |
|------|------|
| Product CLI | `scripts/process.ps1`, `scripts/process_batch.py` |
| Infer | `eval/scripts/run_uie_look.py` |
| Spectroformer repo | `eval/repos/spectroformer/` |
| Current ship weights | `eval/repos/spectroformer/checkpoints/best.pth` |
| NU2Net weights | `eval/repos/uie_benchmark/checkpoints/UIEB/NU2Net.ckpt` |
| Ucolor-PT (parked) | `eval/repos/ucolor_pytorch/` + `checkpoints/best.pth` |
| Credits | `THIRD_PARTY.md` |

## Build adapt pairs from NEFs

Use product decode (embedded JPEG):

```powershell
# From repo root, Windows .venv
.\.venv\Scripts\python.exe -c "
from pathlib import Path
from underwater_pipeline import load_source
from PIL import Image
# ... write RGB PNGs into adapt/train/a/
"
```

Run under `scripts/` on `PYTHONPATH`, or `cd scripts` then import.

Targets in `b/` must be prepared separately (edits or chosen pseudo-GT).

## Spectroformer Train.py notes

Documented layout (`Readme.md`):

```
uw_data/train/a  uw_data/train/b
uw_data/test/a   uw_data/test/b
```

Useful flags (see `Train.py`): `--finetune`, `--epoch_count`, `--lr`,
`--niter`, `--dataset`, `--batch_size`.

Fine-tune load pattern expects a pickled model at
`checkpoint/uw_data/netG_model_epoch_{epoch}.pth` when `--finetune` is true —
align paths with whatever checkpoint naming the run actually writes, or load
`best.pth` explicitly after fixing the script.

Inference today:

```python
# eval/scripts/run_uie_look.py
sys.path.insert(0, str(SPECTRO_ROOT))
model = torch.load(SPECTRO_CKPT, map_location=device, weights_only=False)
```

After adapt, either overwrite `best.pth` (keep a backup) or change `SPECTRO_CKPT`
to `best_adapt_<id>.pth`.

## Suggested fine-tune hyperparams (starting point)

| Knob | Start |
|------|--------|
| Init | current `best.pth` |
| LR | ≤ 3e-5 (upstream default) or lower (1e-5) |
| Batch | 1–2 (VRAM) |
| Epochs / iters | short; stop on val plateau |
| Wall cap | 2–6 h on 5080-class GPU |
| Patch / size | match upstream train crops |

Log: wall time, peak VRAM, val PSNR/SSIM if available, A/B notes.

## A/B protocol

1. Hold out ≥8 NEFs **not** in train pairs (new domain).
2. Run old spectroformer → folder A; new weights → folder B; vivid optional.
3. Human score: cast, haze, artifacts, detail, overall.
4. Ship if new ≥ old on ≥5/8 and no worse artifacts.

## Ucolor-PT reminder

Train: `config.yml` + `python train.py` in `ucolor_pytorch` under `uw_eval`,
fp16, bs=4, Flip→HorizontalFlip patch already applied. Needs `depth/`
transmission triplets. Infer still tiled at 1536 — seams were why it was
parked. Only reopen if pursuing Ucolor look without Podman.
