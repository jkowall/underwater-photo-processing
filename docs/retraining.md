# Retraining / domain adaptation

Normal use is **inference only**. `process.ps1` never trains. It loads published
Spectroformer (or NU2Net) weights and enhances each NEF.

Retrain only when a **new trip/camera/water type** systematically looks worse
than the current default, and classical `-Look vivid` / `-Look nu2net` are not
enough. That is occasional domain adaptation — not something you do per batch.

## How the pieces fit

```text
Day-to-day
  NEFs → process.ps1 → spectroformer weights → PNGs

Rare adapt
  new-domain NEFs + target pairs
       → fine-tune from best.pth (WSL uw_eval)
       → A/B vs old weights
       → if ≥5/8 win, point run_uie_look.py at new checkpoint
```

You do **not** need ground-truth for every photo you process. You only need
paired data for the small adapt set used to fine-tune.

## What “pairs” means

Supervised UIE training needs matching filenames:

| Folder | Content |
|--------|---------|
| `a/` | Degraded inputs (e.g. embedded JPEG from NEF via `load_source`) |
| `b/` | Targets you want (human edits, or carefully chosen pseudo-GT) |

Typical size: tens to a few hundred train pairs + a held-out test set from the
**new** domain. Building good `b/` images is the hard part — not the trainer.

## Where the instructions live

Detailed agent checklist (go/no-go, Spectroformer fine-tune flags, A/B gate,
wiring into the CLI):

- [`.cursor/skills/uie-retrain/SKILL.md`](../.cursor/skills/uie-retrain/SKILL.md)
- [`.cursor/skills/uie-retrain/reference.md`](../.cursor/skills/uie-retrain/reference.md)

Say in chat: “domain-adapt Spectroformer for trip X” and follow that skill.

## What we did *not* ship as the product train loop

An experimental **Ucolor PyTorch** train (UIEB-style triplets) was run once and
**parked** (tile seams; lost A/B vs TF1 Ucolor). Product default remains
**Spectroformer’s published checkpoint**, not a model you must retrain for Day4.

## Licensing

Fine-tune privately. Spectroformer upstream currently has **no LICENSE file** —
do not publish derived weights without author permission. See
[THIRD_PARTY.md](../THIRD_PARTY.md).
