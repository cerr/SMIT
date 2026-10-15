import os
import numpy as np
import sys
import argparse
import nibabel as nib
import SimpleITK as sitk
from skimage.measure import label
import scipy.ndimage.morphology as snm

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from monai import data, transforms
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import Flip, MapTransform
from monai.inferers import sliding_window_inference

from smit_models import configs_smit, smit, smit_plus
from utils.gen_data_json import main as gen_data_json


NETWORK_CONFIGS = {
    'CT_HeartSubstruct_SMIT':  dict(depths=(2, 2, 12, 2), use_encoder5=False),
    'CT_HN_SMIT':     dict(depths=(2, 2, 8, 2),  use_encoder5=False, feature_size=128),
    'CT_Lung_SMIT':   dict(depths=(2, 2, 8, 2),  use_encoder5=False),
    'MR_Rectum_GTV_SMIT': dict(depths=(2, 2, 12, 2), use_encoder5=True),
}

MODEL_LIST = list(NETWORK_CONFIGS.keys())

def list_of_strings(arg):
    """Parse comma-separated string into list."""
    return arg.split(',')


def setup_argparser():
    """Setup argument parser with all configuration options."""
    """Setup argument parser with all configuration options."""
    parser = argparse.ArgumentParser(description="Unified SMIT segmentation pipeline with optional TTA and distributed inference")

    # Data arguments
    parser.add_argument("--data_dir", default=None, type=str, help="dataset directory")
    parser.add_argument('--json_list', default='/scratch/input/data.json',
                        type=str, help='dataset json file')
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
    parser.add_argument("--model", required=True, choices=MODEL_LIST,help="See MODEL_LIST.")
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="path to pretrained model checkpoint")
    parser.add_argument('--model_feature', default=96, type=int, help='model_embedding_feature size')
    parser.add_argument("--amp", action="store_true")
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
    tlist = [transforms.LoadImaged(keys=["image"]),
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


# Load model on device
def load_model(args, device):
    cfg = dict(NETWORK_CONFIGS[args.model])
    if args.use_smit:
        decoder_type = 'upernet' if args.use_upernet else 'unetr'
        config = configs_smit.get_SMIT_128_bias_True(**cfg,
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
            feature_size=cfg['feature_size'],  # or add as arg
            norm_name=args.norm_name,
        )

    checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
    model_dict = checkpoint.get("state_dict", checkpoint)

    # Handle DDP state dict
    new_state_dict = {}
    for k, v in model_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

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
                                  num_workers=args.workers, pin_memory=True)
    loader_iter = tqdm(test_loader, desc=dataset_name) if is_main_process(args) else test_loader
    status = 0

    with torch.no_grad():
        for batch in loader_iter:
            img_path = batch["image_meta_dict"]["filename_or_obj"][0]
            try:
                val_inputs = batch["image"].to(device)
                with autocast(enabled=args.amp and device.type == 'cuda'):
                    batch["pred"] = run_inference_with_tta(model, val_inputs, args)

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
    args = parser.parse_args()

    # Model path
    args.pretrained_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'model_final.pt')

    # Generate data JSON
    gen_data_json(args.data_dir)

    # Setup distributed processing
    num_workers=args.workers
    setup_distributed(args)

    # Set device
    if args.distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print("=" * 60 + "\n")

    if is_main_process(args):
        print("Loading model...")
    model = load_model(args, device)
    test_transform, post_transforms = create_transforms(args)
    for n in args.datasets:
        status = process_dataset(n, args, model, test_transform, post_transforms, device)
        sys.exit(1) if status<0

    cleanup_distributed()

    if is_main_process(args):
        print("Inference completed successfully!")


if __name__ == '__main__':
    main()
