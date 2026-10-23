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
from monai.data import load_decathlon_datalist, decollate_batch, MetaTensor
from monai.transforms import MapTransform
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
    if 'percentile_rescale' in pre:
        d['percentile_rescale'] = pre['percentile_rescale']
    if 'intensity' in pre:
        d['intensity'] = pre['intensity']
    if 'intensity_percentiles' in pre:
        d['intensity_percentiles'] = list(pre['intensity_percentiles'])
    inf = e.get('inference', {})
    for key, arg in (('blend', 'blend'), ('overlap', 'infer_overlap'), ('sw_batch_size', 'sw_batch_size'),
                     ('amp', 'amp'), ('body_mask', 'body_mask'), ('tta', 'use_tta'), ('pad_mode', 'pad_mode')):
        if key in inf:
            d[arg] = inf[key]
    if 'roi' in inf:
        d['roi_x'], d['roi_y'], d['roi_z'] = inf['roi']
    post = e.get('postprocessing', {})
    if 'largest_component' in post:
        d['largest_component'] = post['largest_component']
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
    parser.add_argument("--intensity", default="window", choices=["window", "nonzero_percentile"],
                        help="window: clip to [a_min, a_max] and scale to [0, 1]; nonzero_percentile: clip to the "
                             "--intensity_percentiles of the non-zero voxels and scale to [0, 1] (MR)")
    parser.add_argument("--intensity_percentiles", default=[1.0, 99.0], type=float, nargs=2,
                        help="lower and upper percentile for --intensity nonzero_percentile")
    parser.add_argument("--largest_component", action="store_true",
                        help="keep only the largest connected component of each label and fill its holes")
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
    """Clip at the given percentile of all voxels and rescale to [0, rescale_max] (int16), as the MR rectum
    model was trained. The result stays a MetaTensor so the later transforms still see the image geometry."""
    def __init__(self, keys, percentile=99.5, rescale_max=1000):
        super().__init__(keys)
        self.percentile = percentile
        self.rescale_max = rescale_max

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            image_np = np.asarray(img.cpu() if torch.is_tensor(img) else img, dtype=np.float32)
            perc_val = np.percentile(image_np, self.percentile)
            image_np = np.clip(image_np, a_min=None, a_max=perc_val)
            out = torch.from_numpy(((image_np / perc_val) * self.rescale_max).astype(np.int16))
            d[key] = MetaTensor(out, meta=img.meta, applied_operations=img.applied_operations) \
                if isinstance(img, MetaTensor) else out
        return d


class NonzeroPercentileNormalize(MapTransform):
    """Clip to the given percentiles of the non-zero voxels and scale to [0, 1] (all-zero images unchanged)."""
    def __init__(self, keys, lower=1.0, upper=99.0):
        super().__init__(keys)
        self.lower, self.upper = lower, upper

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            nz = img[img > 0]
            if nz.numel() == 0:
                continue
            lo, hi = (float(v) for v in np.percentile(np.asarray(nz.cpu()), [self.lower, self.upper]))
            d[key] = (img.clamp(min=lo, max=hi) - lo) / max(hi - lo, 1e-8)
        return d


def keep_largest_component(seg_np):
    """For each label: keep the largest 3D connected component (26-connectivity) and fill its holes."""
    out = np.zeros_like(seg_np)
    for lab in np.unique(seg_np):
        if lab == 0:
            continue
        cc, n = label(seg_np == lab, connectivity=3, return_num=True)
        if n == 0:
            continue
        largest = np.argmax(np.bincount(cc.ravel())[1:]) + 1
        out[snm.binary_fill_holes(cc == largest) & (out == 0)] = lab
    return out


def create_transforms(args):
    # image_only=False keeps image_meta_dict (MONAI >= 1.3 defaults to image_only=True)
    tlist = [transforms.LoadImaged(keys=["image"], image_only=False),
             transforms.EnsureChannelFirstd(keys=["image"])]
    if args.percentile_rescale:
        tlist.append(PercentileClipRescale(keys=["image"]))
    if args.intensity == "nonzero_percentile":  # on the original grid, before resampling
        tlist.append(NonzeroPercentileNormalize(keys=["image"], lower=args.intensity_percentiles[0],
                                                upper=args.intensity_percentiles[1]))
    tlist.append(transforms.Spacingd(keys=["image"],
                                     pixdim=(args.space_x, args.space_y, args.space_z),
                                     mode="bilinear"))
    if not args.skip_orientation:
        tlist.append(transforms.Orientationd(keys=["image"], axcodes="RAS"))
    if args.intensity == "window":
        tlist.append(transforms.ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                                     b_min=args.b_min, b_max=args.b_max, clip=True))
    tlist.append(transforms.CropForegroundd(keys=["image"], source_key="image"))
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
                           meta_key_postfix="meta_dict", nearest_interp=False, to_tensor=True),
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


def label_dtype(max_label):
    """Smallest unsigned integer type that holds label values 0..max_label (uint8 up to 255 labels)."""
    for dt in (np.uint8, np.uint16, np.uint32):
        if max_label <= np.iinfo(dt).max:
            return dt
    return np.uint64


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


class UpConv3d(torch.nn.Module):
    """A ConvTranspose3d with kernel == stride and no padding, written as a 1x1x1 convolution to
    out_channels * k^3 channels followed by a voxel shuffle. Same result; unlike ConvTranspose3d it also
    runs in float16 on Apple GPUs (MPS)."""
    def __init__(self, ct):
        super().__init__()
        self.cout, self.k = ct.out_channels, tuple(ct.kernel_size)
        kd, kh, kw = self.k
        self.conv = torch.nn.Conv3d(ct.in_channels, self.cout * kd * kh * kw, 1, bias=False)
        w = ct.weight.detach()  # (in, out, kd, kh, kw) -> (out*kd*kh*kw, in, 1, 1, 1)
        self.conv.weight.data = w.permute(1, 2, 3, 4, 0).reshape(-1, ct.in_channels, 1, 1, 1).clone()
        self.bias = None if ct.bias is None else torch.nn.Parameter(ct.bias.detach().clone())

    @staticmethod
    def applies_to(ct):
        return (tuple(ct.stride) == tuple(ct.kernel_size) and not any(ct.padding) and not any(ct.output_padding)
                and ct.groups == 1 and all(d == 1 for d in ct.dilation))

    def forward(self, x):
        (kd, kh, kw), (n, _, d, h, w) = self.k, x.shape
        y = self.conv(x).view(n, self.cout, kd, kh, kw, d, h, w)
        y = y.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(n, self.cout, d * kd, h * kh, w * kw)
        return y if self.bias is None else y + self.bias.view(1, -1, 1, 1, 1).to(y.dtype)


class Float16Model(torch.nn.Module):
    """Apple GPU (MPS): the whole model in float16 instead of float16 autocast. Autocast keeps float32 weights
    plus float16 copies and, on MPS, needs the transposed convolutions in float32; for the large SMIT+ models
    that peaked at 18 GB (heavy swapping on a 16 GB Mac). In float16: ~5 GB less, ~2.4x faster, same labels.
    Transposed convolutions are rewritten (UpConv3d), normalization layers stay float32. Takes and returns
    float32 tensors."""
    def __init__(self, model):
        super().__init__()
        self.n_upconv = self.n_norm = self.n_fp32 = 0
        for parent in list(model.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.ConvTranspose3d) and UpConv3d.applies_to(child):
                    setattr(parent, name, UpConv3d(child))
                    self.n_upconv += 1
        model.half()
        keep_fp32 = (torch.nn.InstanceNorm3d, torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm3d,
                     torch.nn.ConvTranspose3d)  # any transposed conv left has no float16 kernel on MPS
        for m in model.modules():
            if isinstance(m, keep_fp32):
                m.float()
                m.forward = (lambda f: (lambda x, *a, **k: f(x.float(), *a, **k).to(x.dtype)))(m.forward)
                if isinstance(m, torch.nn.ConvTranspose3d):
                    self.n_fp32 += 1
                else:
                    self.n_norm += 1
        self.model = model

    def forward(self, x):
        return self.model(x.half()).float()


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

    # TTA: average with a left-right flip. MONAI's Flip(spatial_axis=1) on a batched [B, C, x, y, z] tensor flips
    # tensor dim 2, which is what the model was validated with; torch.flip does the same without MONAI's
    # affine bookkeeping (which fails on batched images in recent MONAI versions).
    flip = lambda t: torch.flip(t, dims=(2,))
    preds = []
    for do_flip in [False, True]:
        aug_inputs = val_inputs

        # Apply flip
        if do_flip:
            aug_inputs = flip(aug_inputs)

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
            pred = flip(pred)

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
                use_autocast = amp_dtype is not None and not getattr(args, 'float16_model', False)
                with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                                    enabled=use_autocast):
                    batch["pred"] = run_inference_with_tta(model, val_inputs, args)
                batch["pred"] = batch["pred"].float()  # post-process in float32 (values unchanged)

                b0 = [post_transforms(i) for i in decollate_batch(batch)][0]
                # label map in the smallest unsigned type for this model's labels (uint8 for up to 255 labels)
                seg_np = apply_postprocessing(b0["pred"].float(), args).cpu().numpy().astype(
                    label_dtype(args.out_channels - 1))
                if args.body_mask:
                    seg_np = apply_body_mask(seg_np, img_path)
                if args.largest_component:
                    seg_np = keep_largest_component(seg_np)

                affine = b0["image_meta_dict"].get("original_affine", np.eye(4))
                out_file = os.path.join(output_directory, make_seg_filename(img_path))
                seg_img = nib.Nifti1Image(seg_np, affine)
                seg_img.set_data_dtype(seg_np.dtype)  # store as that integer type, without scaling
                nib.save(seg_img, out_file)
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
    args.float16_model = device.type == 'mps' and get_amp_dtype(args, device) is not None
    if args.float16_model:
        model = Float16Model(model).to(device)
        if is_main_process(args):
            print(f"Apple GPU: model runs in float16 ({model.n_upconv} transposed convolutions rewritten, "
                  f"{model.n_norm} normalization layers in float32"
                  + (f", {model.n_fp32} transposed convolutions in float32" if model.n_fp32 else "") + ")")
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
