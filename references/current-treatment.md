# Current treatment and observed feedback

Updated: 2026-09-23. Classical OpenCV recipe notes; CLI default look is separate.

## CLI default vs classical recipe

- **Product CLI default** (`process.ps1` / `process.sh`): **`auto`**
  (spectroformer underwater, natural topside). See
  [docs/neural-setup.md](../docs/neural-setup.md). Use `-Look spectroformer` /
  `--look spectroformer` for UW-only batches. Neural backends: Windows CUDA,
  Apple Silicon MPS, or WSL fallback.
- **Approved neural finish (Phase A, `phase-a-v1`)** as of 2026-09-23:
  spectroformer / nu2net only. Mild pre-denoise at long-edge 2048 → model →
  guided upsample (source as edge guide) → L detail restore → light
  `polish_neural` (particles + reduced clarity/vibrance) → `reduce_green_cast`.
  Classical looks are unchanged. `recipe_id` includes `neural_finish=phase-a-v1`
  so soft/legacy neural PNGs never resume. Maintainer A/B of the full Day4 set
  preferred Phase A over the prior sharp spectroformer path. Phase B (ISO denoise
  + L contrast + protect) and Phase C (4096 tiled) were gate-tested and rejected
  vs Phase A — keep `phase-a` / `phase-b-gate` / `phase-c-gate-max` folders for
  comparison only.
- **Phase E deferred experiment:** LibRaw/sensor demosaic when embedded JPEG red
  is the ceiling; keep JPEG path as fallback. Not started.
- **Classical OpenCV default** when using `-Look vivid`: the V4 vivid treatment
  below. Still the approved non-neural recipe for CPU-only or fallback runs.

## Approved classical baseline (vivid)

The V4 vivid treatment is the maintainer-approved **classical** recipe as of
2026-09-11. It keeps V2 cleanup and V3 chroma, then reduces residual underwater
green cast. Earlier presets and `work/batch_output_v3` remain available for
comparison. Processed batches belong under gitignored `work/`.

## Source observations

Initial validation used 26 Nikon Z8 NEFs. Source photos and local paths are not
distributed with this repository. Supply your own input and output paths.

All observed embedded JPEGs were 8256 x 5504, matching the RAW crop. Nikon Z8
sensor data failed with rawpy 0.27.1 / LibRaw 0.22.1 and macOS `sips`; do not
assume all future NEFs share this failure. The example embedded JPEG had 16.05%
zero red pixels. Other files had larger deficits, making recovery uncertain.
The camera JPEG profile was assumed sRGB where no embedded profile was present.

The current runner intentionally uses the tested embedded-JPEG path. It rejects
smaller previews and rotated RAW orientation rather than silently upscaling or
misorienting output. Actual sensor-RAW development is a future enhancement.

## Recipes implemented in the helper

Initial correction estimates low-texture upper-corner ambient color, with the
veil capped by 7% of that estimate and 38% of the per-channel fifth percentile.
Soft subtraction avoids hard shadow clipping. Adaptive red supplementation is
0.28 times the surviving green/red difference, reduced in highlights. Trimmed
Shades of Gray uses p=4 in linearized RGB. LAB L-channel CLAHE uses clip limit
1.6, grid 12 x 8, blended at 50%. Clipped cyan highlights receive chroma
suppression to prevent pink patches. Common-channel highlight rolloff avoids
individual channel clipping.

Second pass identifies small compact bright residuals at half resolution,
limited to smooth backgrounds away from detected edges; accepted masks are
expanded to full resolution and interpolated with Telea radius 3. Cleanup
changed less than 0.09% of pixels in any initial batch image. This is an
algorithmic candidate count, not a guarantee every detection was a particle.
It leaves large diffuse particles and uncertain subject-adjacent structures.
Luminance bilateral filtering uses diameter 5, sigmaColor 1.6 in LAB L units,
sigmaSpace 2; chroma uses diameter 7, sigmaColor 3.5, sigmaSpace 3. A modest
adaptive black offset, S curve, and thresholded local contrast add definition.
Its vibrance multiplier `1 + 0.24 * exp(-C/35)` is the subtler `pop` option.

Third pass increases LAB chroma by
`1 + 0.95 * (1 - exp(-(C/6)^2)) * exp(-C/120)`, tapering for bright near-neutral
whites. Residual green (`a < 0`) is amplified at 40% of that gain so vivid does
not reintroduce water cast. Gamut/highlight handling follows.

Fourth pass (`reduce_green_cast`) measures remaining cast from bright
near-neutrals (midtone neutrals if whites are scarce), shifts toward a mild
magenta-neutral white target (`a ≈ 2.2`), warms olive midtones toward brown, and
rotates lime HSV hues toward golden yellow. Strength scales with measured green
in neutrals so already-balanced frames are nudged lightly.

Maintainer review on 2026-09-11 preferred V4 over V3 for residual green on
representative frames (DSC_1565, DSC_1511, DSC_1578, DSC_1539). Keep V3 outputs
for comparison unless the user requests removal.

## How to update this reference

Use V4 vivid as the approved baseline. If the user requests a subtler/stronger
look or different cleanup, revise the helper, validate representative images,
and update these settings. Do not claim visual approval based solely on
numerical checks. Keep earlier outputs available for comparison unless the user
requests removal.
