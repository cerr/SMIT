"""DICOM input / RTSTRUCT output for run_segmentation.py, using pyCERR (core library).

Input layout for --data_dir with DICOM:
  * a folder that directly contains one patient's DICOM files, or
  * a folder with one sub-folder of DICOM files per patient / study.
Each folder is loaded with pyCERR; the first scan of the model's modality (CT or MR) is written
to NIfTI and segmented, and the result is also written as a DICOM RTSTRUCT that references
that scan, so it can be imported alongside the original images.
"""
import contextlib
import io
import os

import numpy as np
import SimpleITK as sitk

DICOM_EXT = ('.dcm', '.ima')


def _import_pycerr():
    try:
        from cerr import plan_container as pc
        from cerr.dcm_export import rtstruct_iod
    except ImportError as e:
        raise SystemExit("DICOM input needs pyCERR: pip install pycerr  (listed in requirements/r2.txt)\n"
                         f"Import error: {e}")
    return pc, rtstruct_iod


def is_dicom_file(path):
    if path.lower().endswith(DICOM_EXT):
        return True
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except OSError:
        return False


def _has_dicom(folder):
    return any(is_dicom_file(os.path.join(folder, n)) for n in os.listdir(folder)
               if os.path.isfile(os.path.join(folder, n)))


def _has_nifti(folder):
    return any(n.lower().endswith(('.nii', '.nii.gz')) for n in os.listdir(folder))


def detect_input_format(data_dir):
    """'nifti' if data_dir holds NIfTI files, 'dicom' if it (or its sub-folders) holds DICOM files."""
    if _has_nifti(data_dir):
        return 'nifti'
    if find_dicom_folders(data_dir):
        return 'dicom'
    raise SystemExit(f"No NIfTI or DICOM files found in {data_dir}")


def find_dicom_folders(data_dir):
    """data_dir itself if it contains DICOM files, else each sub-folder (searched recursively) that does."""
    if _has_dicom(data_dir):
        return [data_dir]
    found = []
    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        if root != data_dir and _has_dicom(root):
            found.append(root)
            dirs[:] = []  # one series folder per patient: do not descend further
    return found


def case_name(folder, data_dir):
    rel = os.path.relpath(folder, data_dir)
    name = os.path.basename(os.path.normpath(folder)) if rel == '.' else rel.replace(os.sep, '_')
    return name


def _scan_type(scan):
    return str(scan.scanInfo[0].imageType).upper()


def dicom_to_nifti(folder, out_nii, modality='CT'):
    """Load a DICOM folder with pyCERR and write the first <modality> scan to NIfTI.

    Returns (planC, scanNum)."""
    pc, _ = _import_pycerr()
    # pyCERR prints patient name and ID while reading; keep them out of logs
    with contextlib.redirect_stdout(io.StringIO()):
        planC = pc.loadDcmDir(folder)
    want = modality.upper()
    matches = [i for i, s in enumerate(planC.scan) if _scan_type(s) in (want, f'{want} SCAN')]
    if not matches:
        found = sorted({_scan_type(s) for s in planC.scan}) or ['none']
        raise ValueError(f"No {modality} scan in {folder} (found: {', '.join(found)})")
    if len(matches) > 1:
        print(f"Note: {len(matches)} {modality} scans in {folder}; using the first one")
    scanNum = matches[0]
    planC.scan[scanNum].saveNii(out_nii)
    return planC, scanNum


def export_rtstruct(seg_nii, planC, scanNum, labels, out_dcm, description):
    """Write the labels present in seg_nii as an RTSTRUCT referencing planC.scan[scanNum].

    labels: {label value: structure name}. Returns the list of structure names written."""
    pc, rtstruct_iod = _import_pycerr()
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_nii))  # a copy: a view would outlive its image
    present = set(np.unique(seg).tolist())
    names = {str(name): int(val) for val, name in labels.items() if int(val) in present}
    if not names:
        print(f"Note: no structures found in {seg_nii}; RTSTRUCT not written")
        return []
    first = len(planC.structure)
    planC = pc.loadNiiStructure(seg_nii, scanNum, planC, names)
    struct_nums = np.arange(first, len(planC.structure))
    rtstruct_iod.create(struct_nums, out_dcm, planC, {'seriesDescription': description})
    return list(names)
