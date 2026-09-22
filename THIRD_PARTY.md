# Third-party notices

This project’s **own** code and documentation are licensed under the
[Apache License 2.0](LICENSE) (Copyright 2026 Jonah Kowall).

Neural looks call **separate upstream projects**. Those components keep their
own copyright and license terms. Apache 2.0 does **not** relicense them.

## Spectroformer (default GPU look)

- **Paper:** Khan et al., “Spectroformer: Multi-Domain Query Cascaded Transformer
  Network for Underwater Image Enhancement,” WACV 2024.
- **Code / checkpoints:** https://github.com/Mdraqibkhan/Spectroformer
- **Authors:** Md Raqib Khan, Priyanka Mishra, Nancy Mehta, Shruti S. Phutke,
  Santosh Kumar Vipparthi, Sukumar Nandi, Subrahmanyam Murala
- **License status (as of 2026-09):** the upstream GitHub repository does **not**
  publish a `LICENSE` file (`license: null` on the API). Do **not** assume you
  may redistribute Spectroformer source or weights under Apache 2.0.
- **This repo’s practice:** treat Spectroformer as an **optional runtime
  dependency** cloned/downloaded locally (under `eval/repos/spectroformer/`,
  gitignored for large assets). Credit the paper when you use or publish results.
  For a public redistribution that **bundles** Spectroformer code or weights,
  obtain explicit permission / a license from the authors first.

```bibtex
@inproceedings{khan2024spectroformer,
  title={Spectroformer: A Multi-Domain Query Cascaded Transformer Network for Underwater Image Enhancement},
  author={Khan, Raqib and Mishra, Priyanka and Mehta, Nancy and Phutke, Shruti S and Vipparthi, Santosh Kumar and Nandi, Sukumar and Murala, Subrahmanyam},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1454--1463},
  year={2024}
}
```

## NU2Net (optional GPU look)

- Used via the [uie_benchmark](https://github.com/ddz16/Underwater-Image-Enhancement-Benchmark) packaging
  (`eval/repos/uie_benchmark/`), which is **MIT** (Copyright 2023 ddz16).
- Model paper / original project: Wu et al., “Underwater Ranker…” (AAAI 2023) —
  https://github.com/RQ-Wu/UnderwaterRanker / https://arxiv.org/abs/2208.06857
- Include the MIT notice when redistributing uie_benchmark code. Confirm
  UnderwaterRanker / checkpoint redistribution terms before shipping weights.

## Classical looks (`vivid` / `pop` / `natural`)

Implemented in this repository (OpenCV / NumPy / rawpy). Covered by **Apache 2.0**.

## What you may distribute on GitHub under Apache 2.0

Safe to push under this project’s Apache 2.0 license:

- `scripts/`, `docs/`, `tests/`, `SKILL.md`, `README.md`, `NOTICE` / this file,
  classical pipeline code, wrappers (`setup.ps1`, `process.ps1`).

Keep out of the public tree (or ship only behind clear upstream terms):

- Vendored `eval/repos/*` clones, especially Spectroformer (unclear license)
- Large `.pth` / `.ckpt` weight files
- Photos / NEFs / generated PNGs

## Training vs inference (practical note)

You do **not** retrain for every dive batch. Pretrained weights are reused for
**inference** on new photos. Retraining is only for rare domain-adaptation work
(new water types / cameras) when quality is systematically poor — and requires
paired ground-truth data. The Day4 Spectroformer run used published weights as-is.
