import os
import json
import warnings
import numpy as np
import sys
import argparse
import nibabel as nib
import SimpleITK as sitk
from skimage.measure import label
import scipy.ndimage as snm

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from monai import data, transforms
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import Flip, MapTransform
from monai.inferers import sliding_window_inference

from smit_models import configs_smit, smit, smit_plus
from utils.gen_data_json import main as gen_data_json
from utils import dicom_io
from model_registry import load_registry, weights_path, missing_weights_message, verify_weights


# Model registry (models.yaml): network settings, preprocessing / inference defaults, labels, weights
MODEL_REGISTRY = load_registry()
MODEL_LIST = list(MODEL_REGISTRY.keys())


def registry_defaults(name):
    """Command-line defaults for a model from models.yaml (options given on the command line still win)."""
    e = MODEL_REGISTRY[name]
    d = {'use_smit': 1 if e['network']['arch'] == 'smit' else 0}
    if 'out_channels' in e:
        d['out_channels'] = e['out_channels']
    pre = e.get('preprocessing', {})
    if 'hu_window' in pre:
        d['a_min'], d['a_max'] = pre['hu_window']
    if 'spacing_mm' in pre:
        d['space_x'], d['space_y'], d['space_z'] = pre['spacing_mm']
    inf = e.get('inference', {})
    for key, arg in (('blend', 'blend'), ('overlap', 'infer_overlap'), ('sw_batch_size', 'sw_batch_size'),
                     ('amp', 'amp'), ('body_mask', 'body_mask')):
        if key in inf:
            d[arg] = inf[key]
    return d


def print_models():
    for name, e in MODEL_REGISTRY.items():
        state = 'installed' if os.path.isfile(weights_path(name)) else 'not installed'
        print(f"{name:28s} {e.get('status', ''):11s} {state:14s} {e.get('description', '')}")
    print(f"\nWeights are installed as {weights_path('<MODEL>')} (see get_weights.py).")


TORCH_REINSTALL = ("uv pip install --reinstall torch==2.8.0 torchvision==0.23.0 "
                   "--index-url https://download.pytorch.org/whl/{}")


def cuda_available():
    """torch.cuda.is_available(), but stop with instructions if an NVIDIA GPU is present and only the
    driver is too old for this PyTorch build (PyTorch would otherwise fall back to the slow CPU silently)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = torch.cuda.is_available()
    if not ok:
        msg = " ".join(str(w.message) for w in caught)
        if "driver" in msg.lower() and "old" in msg.lower():
            sys.exit(f"The NVIDIA driver is too old for this PyTorch build (CUDA {torch.version.cuda}). Update the "
                     f"driver (570 or newer for the CUDA 12.8 build), or install a build for older drivers:\n  "
                     + TORCH_REINSTALL.format("cu126") + "\nPyTorch said: " + msg.strip())
    return ok


def check_cuda_build(index=0):
    """Stop with instructions if the installed PyTorch has no kernels for this GPU (e.g. a Blackwell
    RTX 50-series card with a build older than CUDA 12.8), instead of PyTorch's 'no kernel image' error."""
    major, minor = torch.cuda.get_device_capability(index)
    cap = major * 10 + minor
    built = torch.cuda.get_arch_list()  # e.g. ['sm_75', 'sm_80', ..., 'sm_120']
    for a in built:
        kind, _, num = a.partition('_')
        num = int(num) if num.isdigit() else -1
        # a kernel binary runs on later GPUs of the same generation; PTX (compute_*) is compiled for newer GPUs
        if (kind == 'sm' and num // 10 == major and num <= cap) or (kind == 'compute' and 0 <= num <= cap):
            return
    name = torch.cuda.get_device_name(index)
    if cap >= 70:
        fix = ("For Blackwell GPUs (RTX 50-series, RTX PRO Blackwell) and other recent cards, install the "
               "CUDA 12.8 build:\n  " + TORCH_REINSTALL.format("cu128"))
    else:
        fix = ("Older GPUs (Maxwell, Tesla P100) need the CUDA 12.6 build:\n  " + TORCH_REINSTALL.format("cu126"))
    sys.exit(f"This GPU ({name}, sm_{cap}) is not supported by the installed PyTorch build "
             f"(CUDA {torch.version.cuda}, built for {', '.join(built) or 'no GPUs'}).\n"
             + fix + "\nor run on the CPU with --device cpu (much slower).")


def select_device(choice):
    """CUDA, Apple-silicon GPU (MPS) or CPU. For MPS, setting PYTORCH_ENABLE_MPS_FALLBACK=1 before
    starting Python lets operations that MPS does not implement run on the CPU."""
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    cuda_ok = cuda_available() if choice in ("auto", "cuda") else False
    if choice == "auto":
        choice = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    if choice == "cuda" and not cuda_ok:
        sys.exit("--device cuda requested but no CUDA GPU is available")
    if choice == "mps" and not mps_ok:
        sys.exit("--device mps requested but the Apple-silicon GPU (MPS) is not available")
    if choice == "cuda":
        check_cuda_build()
    return torch.device(choice)


def prepare_dicom_input(args):
    """For DICOM input, convert each series to NIfTI with pyCERR in <results_dir>/dicom_input_nifti and
    point args.data_dir there. Returns {case name: (planC, scanNum)} for the RTSTRUCT export."""
    fmt = args.input_format if args.input_format != 'auto' else dicom_io.detect_input_format(args.data_dir)
    if fmt != 'dicom':
        return {}
    if args.distributed:
        sys.exit("DICOM input is not supported with --distributed; run without it or convert to NIfTI first")
    modality = MODEL_REGISTRY[args.model].get('modality', 'CT')
    stage = os.path.join(args.results_dir, 'dicom_input_nifti')
    os.makedirs(stage, exist_ok=True)
    cases = {}
    for folder in dicom_io.find_dicom_folders(args.data_dir):
        name = dicom_io.case_name(folder, args.data_dir)
        nii = os.path.join(stage, f'{name}.nii.gz')
        print(f'Reading DICOM {folder}')
        try:
            cases[name] = dicom_io.dicom_to_nifti(folder, nii, modality)
        except Exception as e:  # keep going with the other patients
            print(f'Skipping {folder}: {e}')
    if not cases:
        sys.exit(f'No usable {modality} DICOM series found in {args.data_dir}')
    args.data_dir, args.json_list = stage, 'data.json'
    return cases


def export_dicom_results(args, dicom_cases):
    """Write an RTSTRUCT per DICOM case, referencing the scan it was segmented from."""
    labels = MODEL_REGISTRY[args.model].get('labels') or \
        {v: f'{args.model}_label{v}' for v in range(1, args.out_channels)}
    for name, (planC, scanNum) in dicom_cases.items():
        seg = os.path.join(args.results_dir, f'{name}_seg.nii.gz')
        if not os.path.exists(seg):
            continue
        out = os.path.join(args.results_dir, f'{name}_{args.model}_RTSTRUCT.dcm')
        print(f'Writing RTSTRUCT for {name} ...', flush=True)
        written = dicom_io.export_rtstruct(seg, planC, scanNum, labels, out,
                                           f'SMIT {args.model} - research use only')
        if written:
            print(f'Wrote {out}: {", ".join(written)}')


def resolve_weights(args):
    """Weights: --pretrained_model_path if given, else the installed weights/<MODEL>.safetensors (or .pt).
    A missing file is an error; the checksum in models.yaml is verified."""
    entry = MODEL_REGISTRY[args.model]
    explicit = args.pretrained_model_path is not None
    path = args.pretrained_model_path if explicit else weights_path(args.model)
    if not os.path.isfile(path):
        sys.exit(f"Weights file not found: {path}" if explicit else missing_weights_message(args.model))
    if not args.skip_checksum and not verify_weights(path, entry, quiet=not is_main_process(args)):
        msg = (f"{path} does not match the {args.model} weights checksum in models.yaml "
               f"(different version, or a corrupted download).")
        if not explicit:
            sys.exit(msg + " Re-install it with get_weights.py, or pass --skip_checksum to run anyway.")
        print("Warning: " + msg)
    return path

def list_of_strings(arg):
    """Parse comma-separated string into list."""
    return arg.split(',')


def setup_argparser():
    """Setup argument parser with all configuration options."""
    """Setup argument parser with all configuration options."""
    parser = argparse.ArgumentParser(description="Unified SMIT segmentation pipeline with optional TTA and distributed inference")

    # Data arguments
    parser.add_argument("--data_dir", default=None, type=str, help="dataset directory")
    parser.add_argument('--json_list', default='data.json', type=str,
                        help='dataset json file in --data_dir (written automatically)')
    parser.add_argument("--datasets", default=['val'], type=list_of_strings,
                        help="comma-separated list of datasets to pull from json file")
    parser.add_argument("--results_dir", default='results', type=str, help="main output directory")
    parser.add_argument("--output_dir", default=None, type=str, help="secondary output directory")

    # Orientation arguments
    parser.add_argument('--RandFlipd_prob', default=0.8, type=float, help='RandFlipd aug probability')
    parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
    parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
    parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
    parser.add_argument("--skip_orientation", action="store_true",
                        help="skip Orientation transform (for datasets already in correct orientation)")
    parser.add_argument("--pad_mode", default=None, choices=['symmetric', 'z_end'])
    parser.add_argument("--percentile_rescale", action="store_true")
    parser.add_argument("--body_mask", action="store_true")

    # Model arguments
    parser.add_argument("--model", choices=MODEL_LIST, help="model name from models.yaml (see --list_models)")
    parser.add_argument("--list_models", action="store_true", help="list the models in models.yaml and exit")
    parser.add_argument("--skip_checksum", action="store_true", help="do not verify the weights checksum")
    parser.add_argument("--input_format", default="auto", choices=["auto", "nifti", "dicom"],
                        help="auto: NIfTI files in --data_dir, else DICOM (a DICOM folder, or one sub-folder per patient)")
    parser.add_argument("--no_rtstruct", action="store_true",
                        help="for DICOM input, do not write the DICOM RTSTRUCT (NIfTI label maps are always written)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="auto = CUDA GPU, else Apple-silicon GPU (mps), else CPU")
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="weights file (.safetensors or .pt) to use instead of the installed weights/<MODEL>.safetensors")
    parser.add_argument('--model_feature', default=96, type=int, help='model_embedding_feature size')
    parser.add_argument("--amp", action="store_true",
                        help="mixed precision (float16) on a CUDA or Apple-silicon (mps) GPU; the CPU always runs float32")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
    parser.add_argument("--use_upernet", action="store_true", help="Use UPERNET decoder")
    parser.add_argument('--use_smit', default=0, type=int, help='use smit model')

    # Preprocessing arguments
    parser.add_argument('--a_min', default=-140, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=260, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')

    # Inference arguments
    parser.add_argument("--blend", default='gaussian', choices=['constant', 'gaussian'])
    parser.add_argument("--infer_overlap", default=0.5, type=float,
                        help="sliding window inference overlap")
    parser.add_argument("--sw_batch_size", default=16, type=int,
                        help="sliding window batch size")
    parser.add_argument("--use_tta", action="store_true",
                        help="use test-time augmentation (horizontal flip)")

    parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
    parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
    parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')
    parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
    parser.add_argument('--res_block', action='store_true', help='use residual blocks')
    parser.add_argument('--conv_block', action='store_true', help='use conv blocks')

    # Post-processing arguments
    parser.add_argument("--postproc", default="standard",
                        choices=['standard', 'conf_threshold'],
                        help="post-processing method: standard (argmax) or conf_threshold")
    parser.add_argument("--conf_threshold", default=0.70, type=float,
                        help="confidence threshold when using conf_threshold post-processing")

    # Distributed arguments
    parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--distributed', action='store_true', help='start distributed training')
    parser.add_argument('--workers', default=8, type=int, help='number of workers')
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--rank", default=0, type=int, help="rank of the process")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for distributed training")
    parser.add_argument("--dist_url", default="env://", type=str, help="url for distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")

    return parser


def setup_distributed(args):
    """Initialize distributed environment."""
    if args.distributed:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            print("Distributed environment variables not set. Using single GPU.")
            args.distributed = False
            return

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        dist.barrier()

        if args.rank == 0:
            print(f"Distributed initialized: world_size={args.world_size}")


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    """Check if this is the main process."""
    return not args.distributed or args.rank == 0

# Pre-processing options
class PercentileClipRescale(MapTransform):
    def __init__(self, keys, percentile=99.5, rescale_max=1000):
        super().__init__(keys)
        self.percentile = percentile
        self.rescale_max = rescale_max

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image_np = d[key].astype(np.float32)
            perc_val = np.percentile(image_np, self.percentile)
            image_np = np.clip(image_np, a_min=None, a_max=perc_val)
            d[key] = ((image_np / perc_val) * self.rescale_max).astype(np.int16)
        return d


def create_transforms(args):
    # image_only=False keeps image_meta_dict (MONAI >= 1.3 defaults to image_only=True)
    tlist = [transforms.LoadImaged(keys=["image"], image_only=False),
             transforms.EnsureChannelFirstd(keys=["image"])]
    if args.percentile_rescale:
        tlist.append(PercentileClipRescale(keys=["image"]))
    tlist.append(transforms.Spacingd(keys=["image"],
                                     pixdim=(args.space_x, args.space_y, args.space_z),
                                     mode="bilinear"))
    if not args.skip_orientation:
        tlist.append(transforms.Orientationd(keys=["image"], axcodes="RAS"))
    tlist += [transforms.ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                              b_min=args.b_min, b_max=args.b_max, clip=True),
              transforms.CropForegroundd(keys=["image"], source_key="image")]
    if args.pad_mode == 'z_end':
        tlist += [transforms.SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, 0)),
                  transforms.SpatialPadd(keys=["image"], spatial_size=(0, 0, args.roi_z), method='end')]
    else:
        tlist.append(transforms.SpatialPadd(keys=["image"],
                                            spatial_size=(args.roi_x, args.roi_y, args.roi_z)))
    tlist.append(transforms.EnsureTyped(keys=["image"]))
    test_transform = transforms.Compose(tlist)

    post_transforms = transforms.Compose([
        transforms.EnsureTyped(keys="pred"),
        transforms.Invertd(keys="pred", transform=test_transform, orig_keys="image",
                           meta_keys="pred_meta_dict", orig_meta_keys="image_meta_dict",
                           meta_key_postfix="meta_dict", nearest_interp=True, to_tensor=True),
    ])
    return test_transform, post_transforms


# Binary mask of the patient
def fg_mask2d(img_2d, thresh):  #
    mask_map = np.float32(img_2d > thresh)

    def getLargestCC(segmentation):  # largest connected components
        labels = label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    if mask_map.max() < 0.999:
        return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = snm.binary_fill_holes(post_mask)
    return fill_mask


def get_body_mask(img, verbose=False, fg_thresh=1e-4):
    fg_mask_vol = np.zeros(img.shape)
    for ii in range(fg_mask_vol.shape[0]):
        if verbose:
            print("doing {} slice".format(ii))
        _fgm = fg_mask2d(img[ii, ...], fg_thresh)

        fg_mask_vol[ii] = _fgm

    return fg_mask_vol


# Zero labels outside the body (CT only). seg_np is in the original voxel order.
def apply_body_mask(seg_np, img_path, thresh=-150):
    img_zyx = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    body_xyz = np.transpose(get_body_mask(img_zyx, fg_thresh=thresh), (2, 1, 0))
    seg_np[body_xyz == 0] = 0
    return seg_np


def apply_postprocessing(pred_t, args):
    if args.postproc == 'standard':
        return torch.argmax(pred_t, dim=0).to(torch.int64)
    probs = torch.softmax(pred_t, dim=0)
    conf, pred_labels = torch.max(probs, dim=0)
    pred_labels[conf < args.conf_threshold] = 0
    return pred_labels.to(torch.int64)


def load_weights_file(path):
    """State dict from a .safetensors file (tensors only, safe to load) or a PyTorch checkpoint (.pt).
    float16 weights are converted to the model's float32 parameters by load_state_dict."""
    if path.lower().endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path, device='cpu')
    try:
        # .pt checkpoints may hold non-tensor objects, which the safe loader rejects; these files
        # are only loaded after their checksum has been verified against models.yaml
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("state_dict", checkpoint)


# Load model on device
def load_model(args, device):
    cfg = MODEL_REGISTRY[args.model]['network']
    if args.use_smit:
        decoder_type = 'upernet' if args.use_upernet else 'unetr'
        smit_cfg = {k: (tuple(cfg[k]) if k == 'depths' else cfg[k])
                    for k in ('depths', 'use_encoder5') if k in cfg}
        config = configs_smit.get_SMIT_128_bias_True(**smit_cfg,
                                                     decoder_type=decoder_type,
                                                     img_size=(args.roi_x, args.roi_y, args.roi_z))
        model = smit.SMIT_3D_Seg(config,
                                 out_channels=args.out_channels,
                                 img_size=(args.roi_x, args.roi_y, args.roi_z),
                                 norm_name=args.norm_name)

    else:
        model = smit_plus.SMIT_Plus(
            img_size=args.roi_x,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=cfg.get('feature_size', args.model_feature),
            norm_name=args.norm_name,
        )

    model_dict = load_weights_file(args.pretrained_model_path)

    # Handle DDP state dict
    new_state_dict = {}
    for k, v in model_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    # Deep-supervision heads (ds_out*) are only used in training; drop them if the model has none
    own_keys = set(model.state_dict().keys())
    new_state_dict = {k: v for k, v in new_state_dict.items()
                      if k in own_keys or not k.startswith('ds_out')}

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model.to(device)

    return model

# Generate segmentation filename from input path
def make_seg_filename(path):
    name = os.path.basename(os.path.normpath(path))
    for ext in ('.nii.gz', '.nii'):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    fname = f"{name}_seg.nii.gz"
    return fname


def mps_amp_compat(model):
    """PyTorch on Apple GPUs (MPS) cannot run 3D transposed convolutions in float16/bfloat16.
    Run those layers in float32 (mixed precision paused around them); everything else keeps float16."""
    def fp32_forward(orig):
        def forward(x, *a, **k):
            with torch.autocast(device_type='mps', enabled=False):
                return orig(x.float(), *a, **k)
        return forward
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.ConvTranspose3d):
            m.forward = fp32_forward(m.forward)
            n += 1
    return n


# Mixed precision: float16 on GPUs (CUDA, Apple MPS). The CPU runs float32: bfloat16 on the CPU was
# measurably less accurate (lowest structure Dice 0.98 vs GPU) and not faster on CPUs without bfloat16 hardware.
def get_amp_dtype(args, device):
    if args.amp and device.type in ('cuda', 'mps'):
        return torch.float16
    return None


# Run inference with optional test-time augmentation
def run_inference_with_tta(model, val_inputs, args):
    if not args.use_tta:
        pred = sliding_window_inference(
            inputs=val_inputs,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
            mode=args.blend,
            device="cpu",
        )
        return pred

    preds = []
    for do_flip in [False, True]:
        aug_inputs = val_inputs

        # Apply flip
        if do_flip:
            aug_inputs = Flip(spatial_axis=1)(aug_inputs)

        # Run inference
        pred = sliding_window_inference(
            inputs=aug_inputs,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
            mode=args.blend,
            device="cpu",
        )

        # Invert flip
        if do_flip:
            pred = Flip(spatial_axis=1)(pred)

        preds.append(pred)

    return torch.mean(torch.stack(preds), dim=0)


# Process a single dataset
def process_dataset(dataset_name, args, model, test_transform, post_transforms, device):
    if is_main_process(args):
        print(f'Working on: {dataset_name}')
    output_directory = args.results_dir
    if is_main_process(args):
        os.makedirs(output_directory, exist_ok=True)
    if args.distributed:
        dist.barrier()

    datalist_json = os.path.join(args.data_dir, args.json_list)
    test_files = load_decathlon_datalist(datalist_json, True, dataset_name,
                                         base_dir=args.data_dir)
    test_ds = data.Dataset(test_files, transform=test_transform)
    sampler = DistributedSampler(test_ds, shuffle=False) if args.distributed else None
    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, sampler=sampler,
                                  num_workers=args.workers, pin_memory=device.type == "cuda")
    loader_iter = tqdm(test_loader, desc=dataset_name) if is_main_process(args) else test_loader
    status = 0

    with torch.no_grad():
        for batch in loader_iter:
            img_path = batch["image_meta_dict"]["filename_or_obj"][0]
            try:
                val_inputs = batch["image"].to(device)
                amp_dtype = get_amp_dtype(args, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                                    enabled=amp_dtype is not None):
                    batch["pred"] = run_inference_with_tta(model, val_inputs, args)
                batch["pred"] = batch["pred"].float()  # post-process in float32 (values unchanged)

                b0 = [post_transforms(i) for i in decollate_batch(batch)][0]
                seg_np = apply_postprocessing(b0["pred"].float(), args).cpu().numpy().astype(np.uint8)
                if args.body_mask:
                    seg_np = apply_body_mask(seg_np, img_path)

                affine = b0["image_meta_dict"].get("original_affine", np.eye(4))
                out_file = os.path.join(output_directory, make_seg_filename(img_path))
                nib.save(nib.Nifti1Image(seg_np, affine), out_file)
            except Exception as e:
                print(f"\n[Rank {args.rank}] Error processing {img_path}: {e}")
                status=-1
    return status


def main():
    parser = setup_argparser()
    known, _ = parser.parse_known_args()
    if known.list_models:
        print_models()
        return
    if known.model is None:
        parser.error("--model is required (see --list_models)")
    parser.set_defaults(**registry_defaults(known.model))
    args = parser.parse_args()
    args.pretrained_model_path = resolve_weights(args)
    dicom_cases = prepare_dicom_input(args)

    # Generate data JSON
    gen_data_json(args.data_dir)

    # Setup distributed processing
    num_workers=args.workers
    setup_distributed(args)

    # Set device
    if args.distributed:
        device = torch.device(f"cuda:{args.local_rank}")
        check_cuda_build(args.local_rank)
    else:
        device = select_device(args.device)

    if is_main_process(args):
        print("=" * 60)
        print("Inference Configuration")
        print("=" * 60)
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
        print(f"\nDevice: {device}")
        print(f"Distributed: {args.distributed}")
        if args.distributed:
            print(f"World size: {args.world_size}")
        print(f"Orientation transform: {'SKIPPED' if args.skip_orientation else 'RAS'}")
        print(f"TTA: {args.use_tta}")
        amp_dtype = get_amp_dtype(args, device)
        print(f"Mixed precision: {str(amp_dtype).replace('torch.', '') if amp_dtype else 'off (float32)'}")
        print("=" * 60 + "\n")

    if is_main_process(args):
        print("Loading model...")
    model = load_model(args, device)
    if device.type == 'mps' and get_amp_dtype(args, device) is not None:
        n = mps_amp_compat(model)
        if is_main_process(args):
            print(f"Apple GPU: {n} transposed-convolution layers run in float32 (no float16 support on MPS)")
    test_transform, post_transforms = create_transforms(args)
    for n in args.datasets:
        status = process_dataset(n, args, model, test_transform, post_transforms, device)
        if status < 0:
            sys.exit(1)

    cleanup_distributed()

    if dicom_cases and not args.no_rtstruct:
        export_dicom_results(args, dicom_cases)

    # Label names for the output label maps, from models.yaml
    labels = MODEL_REGISTRY[args.model].get('labels')
    if labels and is_main_process(args):
        with open(os.path.join(args.results_dir, 'labels.json'), 'w') as f:
            json.dump({'model': args.model, 'labels': {str(k): v for k, v in labels.items()}}, f, indent=2)

    if is_main_process(args):
        print("Inference completed successfully!")


if __name__ == '__main__':
    main()
