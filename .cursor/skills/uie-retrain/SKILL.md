---
name: uie-retrain
description: >-
  Fine-tune or retrain underwater image enhancement (Spectroformer / NU2Net /
  parked Ucolor-PT) when new dive photos systematically look worse than the
  current look. Use when the user asks to retrain, fine-tune, domain-adapt,
  or improve GPU UIE weights for a new camera, water type, or trip.
---

# UIE retrain / domain adapt

Retrain is **occasional domain adaptation**, not a per-batch step. Prefer
inference with existing weights first. Classical `-Look vivid` remains a
no-retrain fallback.

Read [THIRD_PARTY.md](../../../THIRD_PARTY.md) before redistributing new weights.
Spectroformer upstream has **no LICENSE file** — fine-tune privately; do not
publish derived weights without author permission.

## When to retrain (go / no-go)

**Do retrain / fine-tune** if, on a new trip or camera:

1. Spectroformer (or current default) is **systematically** worse on ≥~30% of
   representative underwater frames vs classical `vivid` or a prior trip, and
2. Spot-checks are not fixed by `-Look nu2net` / `-Look vivid` / `-Look auto`, and
3. You can build a **paired** adaptation set (degraded → target), even if small.

**Do not retrain** for one-off bad frames, topside photos (`auto` → natural),
or mild cast differences. Process with inference only.

## Required data (hard requirement)

Spectroformer / Ucolor supervised training need **pairs**:

```
adapt/
  train/
    a/   # degraded inputs (embedded JPEG / RGB from NEF via load_source)
    b/   # targets (same stem) — expert-edited or carefully chosen pseudo-GT
  test/
    a/
    b/
```

Same basename in `a/` and `b/`. Prefer 50–200+ train pairs; hold out ≥20 test
pairs from the **new** domain. Long-edge ~512–1536 for training patches/resizes.

**Target construction (pick one, document which):**

| Method | Quality | Notes |
|--------|---------|--------|
| Human edit (LR / PS) | Best | Slow; gold standard |
| Prior good look as target | OK | e.g. TF1 Ucolor or vivid on frames where it wins |
| Pseudo-label from another UIE | Risky | Can bake in artifacts; A/B carefully |

Never invent “targets” by copying inputs. Never train on topside sunset as
underwater GT.

## Environment

- WSL2 + micromamba **`uw_eval`** (torch+CUDA), RTX-class GPU
- Repo root: `C:\Users\jkowa\underwater-photo-processing` → `/mnt/c/Users/jkowa/underwater-photo-processing`
- Weights live under `eval/repos/.../checkpoints/` (gitignored)

## Preferred path: Spectroformer fine-tune

Upstream: `eval/repos/spectroformer/` ([WACV 2024](https://github.com/Mdraqibkhan/Spectroformer)).
`Train.py` supports `--finetune` loading a pickled generator.

1. Stage pairs into `uw_data/train/{a,b}` and `uw_data/test/{a,b}` (see upstream Readme).
2. Start from current ship weight: `checkpoints/best.pth` (full pickled module;
   inference also needs repo on `PYTHONPATH` — see `eval/scripts/run_uie_look.py`).
3. Fine-tune with **low LR**, short schedule (hours, not days). Cap wall time;
   keep best val checkpoint.
4. Export / copy winning weights to a **versioned** path, e.g.
   `eval/repos/spectroformer/checkpoints/best_adapt_<trip>.pth`
5. Point product infer at the new file (see [reference.md](reference.md)).
6. **A/B gate:** run `process.ps1` on ≥8 held-out NEFs vs old spectroformer +
   vivid. Ship only if new weights win on ≥5/8 underwater stills **and** no
   worse artifacts. Else keep previous `best.pth`.

If upstream `Train.py` import paths are broken (`final_model` vs
`Final_model_AGSSF`), fix imports before a long run — do not invent a new
architecture in-skill.

## Alternate paths

- **NU2Net:** only if Spectroformer fine-tune fails and nu2net is closer on the
  new domain; use `uie_benchmark` train scripts + MIT attribution.
- **Ucolor-PT (parked):** `eval/repos/ucolor_pytorch/` — already has a working
  train loop (fp16, bs=4, UIEB-style triplets). Prefer only if pursuing a
  Podman-free Ucolor look again; still needs transmission maps at infer.
  See past Agent Store plan `ucolor-pytorch-fork-ab.md`.

## Wire into the product CLI

After a passing A/B:

1. Update the checkpoint path used by `eval/scripts/run_uie_look.py`
   (Spectroformer `SPECTRO_CKPT` or a CLI flag).
2. Smoke one NEF via `.\scripts\process.ps1`.
3. Note trip id, data recipe, epochs, LR, val metric, and A/B verdict in
   `references/uie-adapt-log.md` (create if missing).
4. Do **not** commit large `.pth` files; document download/copy steps.

## Agent checklist

```
- [ ] Confirmed systematic failure (not one-offs)
- [ ] Tried nu2net / vivid / auto first
- [ ] Paired adapt set staged; stems match
- [ ] Fine-tune from current best.pth; log run
- [ ] A/B ≥5/8 vs previous default
- [ ] Versioned weights + adapt-log entry
- [ ] Product path points at new weights; smoke OK
- [ ] Licensing / THIRD_PARTY respected
```

## Details

- Paths, Train.py flags, and wiring snippets: [reference.md](reference.md)
