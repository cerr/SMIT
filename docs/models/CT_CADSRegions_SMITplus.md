# CT_CADSRegions_SMITplus

General anatomical regions (CADS body-region group) on CT and cone-beam CT (CBCT).

| | |
|---|---|
| Architecture | SMIT_Plus, feature size 128, depths 2-2-40-4 |
| Output labels | 1-9 body regions (names unconfirmed, see below), 10 spinal canal / spinal cord |
| Intensity window | -750 to 1750 HU, scaled to [0, 1] |
| Resampling | 1.5 x 1.5 x 1.5 mm, RAS orientation |
| Inference | 128^3 patches, overlap 0.5, Gaussian blending, mixed precision; labels outside the body (-150 HU) removed |
| Weights | `weights/CT_CADSRegions_SMITplus.pt`, installed with `get_weights.py` (checksum in `models.yaml`) |

## Notes

- Only label 10 (spinal canal / cord) has been reviewed; it was visually good on 7 CBCT cases.
- Names of labels 1-9 were inferred from position and volume: subcutaneous tissue, muscle, abdominal cavity,
  thoracic cavity, bones, glands, pericardium, breast implant, mediastinum. They are listed in `models.yaml`
  with an `Unconfirmed_` prefix until the model owner confirms them.
- The checkpoint contains deep-supervision heads (`ds_out*`) that are used only in training; they are skipped
  when loading.

## To be completed by the model owner

Label names, training data, intended use, version history.
