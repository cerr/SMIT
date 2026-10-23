# MR_Rectum_GTV_SMIT

Rectal cancer GTV on T2-weighted MR. Originally distributed as github.com/cerr/MR_Rectum_GTV_SMIT (version 3.0).

| | |
|---|---|
| Architecture | SMIT (SMIT_3D_Seg), embedding 48, depths 2-2-12-2, fifth encoder block, UNETR decoder |
| Output labels | 1 GTV |
| Intensity | clipped at the 99.5th percentile and rescaled to 0-1000 (on the original grid), then window 0 to 800 scaled to [0, 1] |
| Resampling | 1.0 x 1.0 x 1.0 mm, RAS orientation |
| Inference | 128 x 128 x 64 patches, padded at the end in z, overlap 0.5, Gaussian blending, sliding-window batch 1, flip test-time augmentation, mixed precision (the original ran float32; 4 voxels differed on the test case, Dice 0.9999) |
| GPU memory | about 2.5 GB peak (Quadro RTX 5000, sliding-window batch 1) |
| Post-processing | largest connected component of the GTV, holes filled (as in pyCERR's MR_Rectum_GTV_SMIT post-processing) |
| Weights | `weights/MR_Rectum_GTV_SMIT.safetensors`, installed with `get_weights.py` (checksum in `models.yaml`) |

## Notes

- Settings reproduce the original repository's `bash_run_SMIT_Segmentation.sh` (with its line-continuation bug
  fixed, which had dropped `--out_channels` and every argument after it) and pyCERR's post-processing. Predictions are mapped back to the original image grid by interpolating the class scores linearly before taking the label (the original used nearest neighbour), which gives smoother boundaries.
- `--largest_component` can be turned off on the command line by overriding the registry, e.g. to inspect the raw
  model output.

## To be completed by the model owner

Training data, intended use, evaluation, version history.
