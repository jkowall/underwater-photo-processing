# Current treatment and observed feedback

Updated: 2026-09-10. This is a maintained working recipe, not a calibrated camera
profile or a universally appropriate underwater look.

## Approved baseline

The V3 vivid treatment, retaining V2 cleanup, is the maintainer-approved default
as of 2026-09-10. It combines lossless PNG output, conservative cleanup, and
visibly stronger color. Earlier presets remain available for a subtler look.

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
0.20 times the surviving green/red difference, reduced in highlights. Trimmed
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

Third pass completed all 26 files (1.62 GB), with zero measured output highlight
clipping and a batch median colored-pixel chroma ratio of 1.79 versus V2. The
standalone skill runner reproduced DSC_1565's V3 output pixels exactly from its
NEF and preserved the capture date and profile. This treatment was visually
approved by the maintainer on 2026-09-10.

Third pass builds on the actual second-pass pixels. It increases LAB chroma by
`1 + 0.95 * (1 - exp(-(C/6)^2)) * exp(-C/120)`, tapering for bright near-neutral
whites and applying gamut/highlight handling afterward. This is approximately
a 70-80% measured chroma increase over V2 in the tested colored regions, rather
than a literal 70-80% saturation-slider setting. Do not boost channel means or
invent object hues simply to make an image colorful. Preserve V2's cleanup.

## How to update this reference

Use V3 vivid as the approved baseline. If the user requests a subtler/stronger look
or different cleanup, revise the helper,
validate representative images, and update these settings. Do not claim visual
approval based solely on numerical checks. Keep earlier outputs available for
comparison unless the user requests removal.
