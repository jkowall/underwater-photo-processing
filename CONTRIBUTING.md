# Contributing

Set up Python 3.12 with `requirements.txt`, then run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

For algorithm changes, compare representative images at equivalent framing and
zoom. Check neutral highlights, subject detail, water backgrounds, and clipping.
Do not commit private photographs or generated results. Synthetic regression
fixtures should be generated in tests; contributors must have sharing rights
for any future sample images.

Keep treatment changes explicit. The vivid recipe is the approved baseline;
setup or reliability improvements should not alter its pixels on valid inputs.
Update `references/current-treatment.md`, relevant docs, and the installed skill
when an intentional new treatment is accepted. Record measured validation
separately from aesthetic approval.

Future improvements worth evaluating include sensor-RAW decoding, color-managed
rendered inputs, selective background masks, and user-reviewable particle edits.
Each needs representative visual validation before becoming the default.

By submitting a contribution, you agree it may be distributed under the project's
Apache 2.0 license. Dependencies keep their own licenses.
