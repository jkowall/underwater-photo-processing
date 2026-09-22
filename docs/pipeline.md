# Processing pipeline

## Source representation

`rawpy` extracts the NEF's embedded camera JPEG. Its dimensions must equal the
RAW crop, and the recorded RAW orientation must be unrotated. This is not a
RAW demosaic. The camera rendering is treated as sRGB, so camera files rendered
in another color space are outside the current tested recipe.

All presets begin with these stages:

1. Invert the sRGB transfer function to work in linearized RGB.
2. Estimate low-texture upper-corner ambient color and a conservative additive
   veil, bounded by the image's dark channel tails. Apply soft subtraction.
3. Estimate missing red from surviving green structure with highlight tapering.
   Apply trimmed Shades of Gray white balance in linearized RGB.
4. Apply CLAHE to LAB luminance with a partial blend. Reduce chroma in clipped
   cyan highlights to avoid artificial pink patches.
5. Convert LAB through XYZ into linear RGB and apply common-channel highlight
   rolloff before encoding sRGB.

`pop` adds small-particle candidate detection, local interpolation, bilateral
denoising, an adaptive black offset, an S curve, and restrained local contrast.
`vivid` adds a stronger LAB chroma increase, tapering for already vivid colors
and bright near-neutral whites. See the [recipe](../references/current-treatment.md)
for exact parameters and initial validation.

GPU looks (`spectroformer`, `nu2net`) skip this classical chain and call
`eval/scripts/run_uie_look.py` via WSL — see [neural-setup.md](neural-setup.md).

## Noise and particulate are different

Bilateral filters reduce fine luminance and chroma noise while limiting blur
across edges. Visible water particles require selective spatial edits. The
detector looks for compact bright residuals in smooth background regions away
from edges and interpolates only accepted masks. The conditions reduce, but do
not eliminate, the risk of confusing a small real feature with a particle.

Broad diffuse backscatter and ambiguous structures near coral, fish patterns,
or appendages are intentionally not aggressively removed. Inspect crops before
raising cleanup strength. Do not interpret candidate counts as a measured
count of actual water particles.

## Color and quality limits

Red values clipped in the camera JPEG cannot be uniquely recovered. Ambient
light and transmission cannot be uniquely inferred from one image either.
The model is physics-inspired but heuristic; it does not solve calibrated
spectral attenuation or estimate measured scene depth.

White balance can neutralize naturally colored subjects or water, and saturation
can amplify a remaining cast. Assess representative scenes and favor selective
changes when a global adjustment harms whites or background water. The approved
vivid look is a preference, not evidence of true subject colors.

PNG is lossless relative to the final 8-bit output pixels. It does not become a
16-bit RAW master or undo JPEG compression. Capture date and selected camera
metadata are retained, while opaque maker notes and GPS tags are not copied.

The runner verifies size, ICC profile, capture date, PNG integrity, exact decoded
pixels, and source/output checksums. Histograms and clipping statistics support
review; they do not replace visual inspection.
