# CT_WHOLEBODY_SMITplus

Whole-body anatomy on CT. Originally distributed as github.com/cerr/CT_WHOLEBODY_SMITplus.

| | |
|---|---|
| Architecture | SMIT_Plus, feature size 128, depths 2-2-40-4 |
| Output labels | 46 structures (names in `models.yaml`; also written to `labels.json` in each results folder) |
| Intensity window | -500 to 500 HU, scaled to [0, 1] |
| Resampling | 1.5 x 1.5 x 2.0 mm, RAS orientation |
| Inference | 128^3 patches, overlap 0.5, constant blending, sliding-window batch 1, mixed precision; labels outside the body (-150 HU) removed |
| GPU memory | about 11.5 GB peak (Quadro RTX 5000, sliding-window batch 1) |
| Weights | `weights/CT_WHOLEBODY_SMITplus.safetensors`, installed with `get_weights.py` (checksum in `models.yaml`) |

## Notes

- Settings reproduce the original repository's `bash_run_SMIT_Segmentation.sh`, which resamples to 2.0 mm in z. Predictions are mapped back to the original image grid by interpolating the class scores linearly before taking the label (the original used nearest neighbour), which gives smoother boundaries.
  The original checkpoint file is named `CT_WHOLEBODY_15_15_15.pt`, which suggests 1.5 mm; the model owner should
  confirm which spacing the model was trained with.

## To be completed by the model owner

Confirmation of the z spacing (1.5 or 2.0 mm), training data, intended use, evaluation, version history.
