# MR_HN_Nodule_SMIT

Head and neck nodule segmentation on MR. Originally distributed as github.com/cerr/MR_HN_Nodule_SMIT.

| | |
|---|---|
| Architecture | SMIT (SMIT_3D_Seg), embedding 48, depths 2-2-8-2, no encoder5, UNETR decoder |
| Output labels | 1 Nodule |
| Intensity | clipped to the 1st-99th percentile of the non-zero voxels and scaled to [0, 1] (the original `preprocess.py`); no further intensity window |
| Resampling | 1.5 x 1.5 x 1.5 mm, RAS orientation |
| Inference | 128^3 patches, overlap 0.5, constant blending, sliding-window batch 1, mixed precision; no body mask |
| GPU memory | about 3 GB peak (Quadro RTX 5000, sliding-window batch 1) |
| Weights | `weights/MR_HN_Nodule_SMIT.safetensors`, installed with `get_weights.py` (checksum in `models.yaml`) |

## Notes

- Predictions are mapped back to the original image grid by interpolating the class scores linearly before taking the label (the original used nearest neighbour), which gives smoother boundaries.
- In the original repository the percentile normalization was a separate step (`preprocess.py`); here it is part of
  the pipeline, so raw MR images are given directly.
- The original repository's `args_to_run.py` is a saved Box web page rather than the arguments file; the settings
  here come from `bash_run_MR_HN_SMIT.sh` and `run_segmentation.py`.
- Not yet evaluated on head and neck MR test data.

## To be completed by the model owner

Training data, intended use, evaluation, version history.
