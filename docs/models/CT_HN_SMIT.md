# CT_HN_SMIT

Head and neck organs at risk on CT. Originally distributed as github.com/cerr/CT_HN_SMIT.

| | |
|---|---|
| Architecture | SMIT_Plus, feature size 128, depths 2-2-40-4 |
| Output labels | 1 Brainstem, 2 Parotid_L, 3 Parotid_R, 4 Submandibular_L, 5 Submandibular_R, 6 Mandible, 7 Oral_Cavity, 8 SpinalCord, 9 Pterygoid_L, 10 Pterygoid_R, 11 Masseter_L, 12 Masseter_R, 13 Thyroid, 14 Larynx, 15 BrachialPlexus_L, 16 BrachialPlexus_R |
| Intensity window | -750 to 1750 HU, scaled to [0, 1] |
| Resampling | 1.5 x 1.5 x 1.5 mm, RAS orientation |
| Inference | 128^3 patches, overlap 0.5, constant blending, sliding-window batch 1, mixed precision; labels outside the body (-150 HU) removed |
| GPU memory | about 9 GB peak (Quadro RTX 5000, sliding-window batch 1) |
| Weights | `weights/CT_HN_SMIT.safetensors`, installed with `get_weights.py` (checksum in `models.yaml`) |

## Notes

- The original run script set `use_smit=1` and then overrode it with `use_smit=0`; the network is SMIT_Plus.
- Settings reproduce the original repository's `bash_run_SMIT_Segmentation.sh`. Predictions are mapped back to the original image grid by interpolating the class scores linearly before taking the label (the original used nearest neighbour), which gives smoother boundaries.

## To be completed by the model owner

Training data, intended use, evaluation, version history.
