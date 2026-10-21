# CT_ThoraxOARs_SMITplus

Thoracic organs at risk on CT and cone-beam CT (CBCT).

| | |
|---|---|
| Architecture | SMIT_Plus, feature size 128, depths 2-2-40-4 |
| Output labels | 1 Lung_L, 2 Lung_R, 3 Esophagus, 4 Heart |
| Intensity window | -750 to 1750 HU, scaled to [0, 1] |
| Resampling | 1.5 x 1.5 x 2.0 mm, RAS orientation |
| Inference | 128^3 patches, overlap 0.5, Gaussian blending, mixed precision; labels outside the body (-150 HU) removed |
| Weights | `weights/CT_ThoraxOARs_SMITplus.safetensors`, installed with `get_weights.py` (checksum in `models.yaml`) |
| GPU memory | about 8 GB peak (Quadro RTX 5000, sliding-window batch 1) |
| Run time | about 45-60 s per CBCT on a Quadro RTX 5000, including loading the model |

## Notes

- Labels 1 and 2 are left and right lung by position in the output; merge them for a single lung structure.
- The network uses the `window_reverse` permutation as shipped in `smit_models/smit_plus.py`; a "corrected"
  permutation does not work with these weights.

## Evaluation (CBCT)

Internal evaluation on HyperSight CBCT scans (0.947 x 0.947 x 2.0 mm), Dice against clinical contours on the
cases whose RTSTRUCT shares the CBCT frame of reference:

| Structure | Cases | Dice |
|---|---|---|
| Esophagus | 4 | 0.32-0.65 (mean 0.50) |
| Heart | 2 | 0.28, 0.85 |

Lungs were visually good on all 7 cases reviewed. Heart is incomplete on CBCTs with strong streak artifacts.

## To be completed by the model owner

Training data, intended use, version history.
