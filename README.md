# SMIT segmentation models

Inference code for 3D medical image segmentation models built on SMIT
(self-distilled masked image transformer) and SMIT_Plus. One code base serves several models;
each model is described in [`models.yaml`](models.yaml) and has a model card in [`docs/models/`](docs/models/).

**Model weights are not part of this repository.** They are shared separately on request
(see [Getting model weights](#getting-model-weights)).

> **Research use only.** This software has not been reviewed or approved by the FDA and must not be used
> in the provision of patient care. See [LICENSE](LICENSE).

## Models

| Model | Modality | Structures | Status |
|---|---|---|---|
| [`CT_ThoraxOARs_SMITplus`](docs/models/CT_ThoraxOARs_SMITplus.md) | CT / CBCT | Left lung, right lung, esophagus, heart | validated |
| [`CT_LungGTV_SMITplus`](docs/models/CT_LungGTV_SMITplus.md) | CT / CBCT | Lung tumor GTV | validated |
| [`CT_CADSRegions_SMITplus`](docs/models/CT_CADSRegions_SMITplus.md) | CT / CBCT | Spinal canal / cord (+ 9 body regions, names unconfirmed) | validated |
| [`CT_HeartSubstruct_SMIT`](docs/models/CT_HeartSubstruct_SMIT.md) | CT | Aorta, pulmonary artery and veins, SVC, IVC, four chambers | validated |
| `CT_HN_SMIT`, `CT_Lung_SMIT`, `MR_Rectum_GTV_SMIT` | | | incomplete: network settings only |

`python run_segmentation.py --list_models` shows the same list and which weights are installed.

## Quick start

From a fresh machine to segmentations, using `CT_ThoraxOARs_SMITplus` as the example. You need an NVIDIA
GPU with a driver that supports CUDA 12 (a CPU works too, much more slowly), git, and the download link for
the model's weights from the maintainers.

**Linux**

```bash
# 1. uv (Python package manager), if not installed yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Code and environment
git clone https://github.com/cerr/SMIT.git ~/SMIT
cd ~/SMIT
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/r1.txt
uv pip install -r requirements/r2.txt

# 3. Model weights -> ~/SMIT/weights/CT_ThoraxOARs_SMITplus.safetensors
python get_weights.py --model CT_ThoraxOARs_SMITplus --url "<link you received>"

# 4a. NIfTI input: a folder of .nii / .nii.gz scans
python run_segmentation.py --model CT_ThoraxOARs_SMITplus \
    --data_dir /data/study/nifti --results_dir /data/study/thorax_results

# 4b. DICOM input: one sub-folder of DICOM files per patient
python run_segmentation.py --model CT_ThoraxOARs_SMITplus \
    --data_dir /data/study/dicom --results_dir /data/study/thorax_results
```

**Windows** (Command Prompt)

```bat
REM 1. uv (Python package manager), if not installed yet
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

REM 2. Code and environment
git clone https://github.com/cerr/SMIT.git C:\tools\SMIT
cd C:\tools\SMIT
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements\r1.txt
uv pip install -r requirements\r2.txt

REM 3. Model weights -> C:\tools\SMIT\weights\CT_ThoraxOARs_SMITplus.safetensors
python get_weights.py --model CT_ThoraxOARs_SMITplus --url "<link you received>"

REM 4a. NIfTI input: a folder of .nii / .nii.gz scans
python run_segmentation.py --model CT_ThoraxOARs_SMITplus --data_dir D:\study\nifti --results_dir D:\study\thorax_results

REM 4b. DICOM input: one sub-folder of DICOM files per patient
python run_segmentation.py --model CT_ThoraxOARs_SMITplus --data_dir D:\study\dicom --results_dir D:\study\thorax_results
```

In a new terminal, activate the environment again (step 2's `activate` line) before running.
On macOS, see [Installation](#installation) for the PyTorch step.

**What you get**, for input scans `pat01`, `pat02`, ...:

```
thorax_results/
├── pat01_seg.nii.gz                              label map on the input scan's grid
├── pat02_seg.nii.gz
├── labels.json                                   1 Lung_L, 2 Lung_R, 3 Esophagus, 4 Heart
│   (DICOM input only:)
├── pat01_CT_ThoraxOARs_SMITplus_RTSTRUCT.dcm     contours referencing the original scan
├── pat02_CT_ThoraxOARs_SMITplus_RTSTRUCT.dcm
└── dicom_input_nifti/                            the scans converted to NIfTI
```

A run takes about a minute per scan per model on a 16 GB workstation GPU. Run another model by installing
its weights (step 3) and changing `--model`; `python run_segmentation.py --list_models` shows what is
available and installed.

## Installation

Requires Python 3.12 and, for GPU inference, an NVIDIA driver that supports CUDA 12.
With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/cerr/SMIT.git
cd SMIT
uv venv --python 3.12
uv pip install -r requirements/r1.txt
uv pip install -r requirements/r2.txt
```

Activate the environment (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows)
before running the commands below.

**NVIDIA GPUs:** `requirements/r1.txt` installs PyTorch's CUDA 12.8 build, which supports GPUs from the
GTX 10-series (Pascal) through Blackwell (RTX 50-series, RTX PRO Blackwell) and needs an NVIDIA driver 570 or
newer. If `run_segmentation.py` reports that the driver is too old or the GPU is not
supported, it prints the command to install a matching build; for reference:

| Situation | PyTorch build |
|---|---|
| Driver 570+ (GTX 10-series through Blackwell) | `cu128` (default) |
| Driver older than 570, or a Maxwell GPU / Tesla P100 | `cu126` |
| No NVIDIA GPU | `cpu` |

For example, with an older driver:

```bash
uv pip install --reinstall torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

**macOS (Apple silicon):** PyTorch has no CUDA builds for macOS, so install it from the default index
instead of `requirements/r1.txt`:

```bash
uv pip install torch==2.8.0 torchvision==0.23.0
uv pip install -r requirements/r2.txt
```

Run with `--device mps` (the default `auto` picks it when no CUDA GPU is present) and set
`PYTORCH_ENABLE_MPS_FALLBACK=1` so that any operation the Apple GPU backend lacks runs on the CPU.
Mixed precision (float16) is used on CUDA and Apple-silicon GPUs for models with `amp: true` in
`models.yaml`; on the CPU the models run in float32. On Apple silicon the results match a CUDA GPU to within
0.001% of voxels. Memory is shared with the rest of the system. With `--sw_batch_size 1` (the SMIT_Plus
models' default; `CT_HeartSubstruct_SMIT` defaults to 4, which needs more), `CT_HeartSubstruct_SMIT` needs about
12 GB and runs in about 1.5 minutes per scan on an M4 with 16 GB, while the SMIT_Plus models (e.g.
`CT_ThoraxOARs_SMITplus`) need about 20 GB, so use a Mac with 24 GB or more for them; with less they run, but
very slowly.

## Getting model weights

Weights are distributed separately from the code. Request access to the models you need from the
maintainers; you will receive a download link per model. Install it with:

```bash
python get_weights.py --model CT_ThoraxOARs_SMITplus --url "<link you received>"
# or, if you already downloaded the file:
python get_weights.py --model CT_ThoraxOARs_SMITplus --file /path/to/download
python get_weights.py --list
```

This installs the weights as `weights/<MODEL>.safetensors` in the repository folder, e.g.
`weights/CT_ThoraxOARs_SMITplus.safetensors`, after checking them against the SHA-256 checksum in `models.yaml`,
so a wrong or corrupted file is rejected. `run_segmentation.py --model <MODEL>` uses that file; if it is
missing, it stops and prints the `get_weights.py` command to run. To use a checkpoint stored elsewhere,
pass `--pretrained_model_path`.

## Running

Put the input images (NIfTI, `.nii` / `.nii.gz`) in one folder and run:

```bash
python run_segmentation.py --model CT_ThoraxOARs_SMITplus --data_dir /path/to/images --results_dir /path/to/output
```

- Every NIfTI file in `--data_dir` is segmented; the list is written to `data.json` in that folder.
- Each output is `<input name>_seg.nii.gz`, a label map on the input image grid, and `labels.json`
  in the results folder maps label values to structure names.
- Preprocessing (intensity window, resampling spacing), sliding-window settings, mixed precision and body
  masking come from the model's entry in `models.yaml`. Any of them can be overridden on the command line,
  e.g. `--a_min`, `--space_z`, `--blend`, `--sw_batch_size`.
- `--device auto` (default) uses a CUDA GPU, else an Apple-silicon GPU (`mps`), else the CPU. The CPU gives the
  same segmentations but is much slower and needs a lot of memory: for `CT_ThoraxOARs_SMITplus`, about
  5 minutes and up to 53 GB of RAM per scan on a 12-core workstation CPU, against under a minute on a GPU.
- `--pretrained_model_path` runs a specific weights file (`.safetensors` or PyTorch `.pt`) instead of the installed one.
- See `python run_segmentation.py --help` for test-time augmentation, confidence thresholding and
  multi-GPU (`torchrun`) options.

### DICOM input

`--data_dir` can also point to DICOM: either one folder of a patient's DICOM files, or a folder with one
sub-folder per patient. The format is detected automatically (`--input_format` forces it). DICOM is read with
[pyCERR](https://github.com/cerr/pyCERR) (installed with `requirements/r2.txt`); for each folder the first scan of the
model's modality is segmented, and the results folder gets

- `<case>_seg.nii.gz` and `labels.json`, as for NIfTI input,
- `<case>_<MODEL>_RTSTRUCT.dcm`, a DICOM RTSTRUCT that references the original scan, with one structure per
  label found (named as in `models.yaml`); `--no_rtstruct` skips it,
- `dicom_input_nifti/`, the scans converted to NIfTI.

The RTSTRUCT carries the patient and study identifiers of the input DICOM, so handle it like the input data.
DICOM input cannot be combined with `--distributed`.

## Adding a model

Add an entry to `models.yaml` with the network settings, preprocessing, inference options and labels,
then install its weights with `get_weights.py --file` and put the printed SHA-256 in the entry.
Add a model card in `docs/models/`.

## Distributing weights (maintainers)

Weights are currently shared on request: send the user a download link and they install it with
`get_weights.py --url <link>`. Every model's `models.yaml` entry holds the SHA-256 checksum, so whatever the
channel, users get exactly the released file.

To publish a model's weights through a download service instead, add a `source` to its `weights` entry; then
`python get_weights.py --model <MODEL>` downloads without a link:

```yaml
weights:
  sha256: "..."
  source: {type: huggingface, repo: cerr/CT_ThoraxOARs_SMITplus, file: CT_ThoraxOARs_SMITplus.safetensors, revision: v1.0}
```

For Hugging Face, a gated repository with manual approval lets you approve each user; access is granted per
repository, so put each model in its own repository to approve models separately, or several models in one
repository to approve them together. Users request access on the repository page and log in with
`hf auth login`. `{type: url, url: ...}` works for any direct download link.

## Citation

If you publish results produced with this software, cite:

- https://doi.org/10.1007/978-3-031-16440-8_53
- https://doi.org/10.1002/mp.70701

## License

Research use only; redistribution outside your organization requires prior written permission from the
CERR development team. See [LICENSE](LICENSE) for the full terms, including export control.
