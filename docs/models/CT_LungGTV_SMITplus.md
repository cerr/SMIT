# CT_LungGTV_SMITplus

Lung tumor gross tumor volume (GTV) on CT and cone-beam CT (CBCT).

| | |
|---|---|
| Architecture | SMIT_Plus, feature size 128, depths 2-2-40-4 |
| Output labels | 1 GTV |
| Intensity window | -500 to 500 HU, scaled to [0, 1] |
| Resampling | 1.5 x 1.5 x 2.0 mm, RAS orientation |
| Inference | 128^3 patches, overlap 0.5, Gaussian blending, mixed precision; labels outside the body (-150 HU) removed |
| Weights | `weights/CT_LungGTV_SMITplus.safetensors`, installed with `get_weights.py` (checksum in `models.yaml`) |
| GPU memory | about 8 GB peak (sliding-window batch 1) |

## Notes

- The intensity window was given by the model owner; the spacing was chosen by a positive control on training cases.

## Evaluation (CBCT)

Internal evaluation on 4 HyperSight CBCT cases, Dice against clinical GTV / iGTV contours: 0.00-0.78 (mean 0.31). (Scored before 2026-10-23, when predictions were mapped back to the image grid with nearest neighbour; current results use linear interpolation and may differ slightly.)
The model gives false positives in consolidated lung and missed one lesion; treat CBCT results with caution.

## To be completed by the model owner

Training data, intended use, version history.
