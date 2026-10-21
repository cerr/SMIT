# CT_HeartSubstruct_SMIT

Cardiac substructures on CT.

| | |
|---|---|
| Architecture | SMIT (SMIT_3D_Seg), embedding 48, depths 2-2-12-2, UNETR decoder |
| Output labels | 1 Aorta, 2 Pulmonary artery, 3 Pulmonary vein, 4 Superior vena cava, 5 Inferior vena cava, 6 Right atrium, 7 Right ventricle, 8 Left atrium, 9 Left ventricle |
| Intensity window | -200 to 300 HU, scaled to [0, 1] |
| Resampling | 1.0 x 1.0 x 3.0 mm, RAS orientation |
| Inference | 128^3 patches, overlap 0.5, constant blending, sliding-window batch 4, mixed precision; labels outside the body (-150 HU) removed |
| Weights | `weights/CT_HeartSubstruct_SMIT.safetensors`, installed with `get_weights.py` (checksum in `models.yaml`) |

## Notes

- Settings match the original CT_HeartSubstruct_SMIT release (`bash_run_SMIT_HeartSubstructures.sh`).

## To be completed by the model owner

Training data, intended use, evaluation, version history.
