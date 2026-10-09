# Spectrum reconstruction development

Read README.md, docs/HISTORY.md and provenance/manifest.json before modifying models.
This Git history reconstructs development stages in 2026; it is not the original dated commit history.
Recovered source snapshots under historical/ are reference-only. Make runtime fixes in root model.py or the shared runner, and record the difference.
The main task is solution absorption-spectrum reconstruction from encoded color-patch RGB, not image hyperspectral reconstruction.
Keep train and prediction preprocessing identical. Respect configured feature layout, scale and wavelength endpoints.
Separate samples by solution/physical capture before augmentation or repeated sampling. Filename checks alone cannot establish absence of leakage.
Run python -m unittest discover -s tests -v and a synthetic train/checkpoint/predict smoke check when changing model or data contracts.
Synthetic checks establish execution only. Never claim real accuracy or an independently replicated experiment without the real data and split provenance.
Do not publish private conversation exports, sample photographs, measured spectra or credentials without explicit authorization.
