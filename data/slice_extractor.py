"""
data/slice_extractor.py
-----------------------
Extracts 2D axial slices from 3D NIfTI CT volumes (LiTS / CHAOS datasets).
Implements clinical Hounsfield Unit (HU) windowing [-100, 400] and intensity normalization.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Clinical HU window for abdominal/liver CT scans
HU_MIN = -100.0
HU_MAX = 400.0
TARGET_SIZE = (256, 256)


def apply_hu_window(volume: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    """
    Clips raw CT Hounsfield Units to the liver-specific window and normalizes to [0, 1].
    
    Args:
        volume: Raw CT volume or slice in Hounsfield Units.
        hu_min: Lower bound for liver soft-tissue window (-100 HU).
        hu_max: Upper bound for liver soft-tissue window (400 HU).
        
    Returns:
        Normalized array in range [0.0, 1.0], float32.
    """
    clipped = np.clip(volume, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


def compute_slice_metadata(mask_slice: np.ndarray) -> Dict[str, Any]:
    """
    Computes clinical lesion and organ statistics for a 2D axial slice.
    
    Mask classes:
      0: Background
      1: Liver Parenchyma
      2: Tumor / Lesion
    """
    liver_pixels = int(np.sum(mask_slice >= 1))
    tumor_pixels = int(np.sum(mask_slice == 2))
    
    tumor_ratio = float(tumor_pixels / liver_pixels) if liver_pixels > 0 else 0.0
    
    if tumor_pixels == 0:
        category = "normal"
    elif tumor_pixels < 250:  # ~small focal nodule
        category = "small_tumor"
    elif tumor_pixels < 1200:
        category = "medium_tumor"
    else:
        category = "large_tumor"
        
    return {
        "liver_pixels": liver_pixels,
        "tumor_pixels": tumor_pixels,
        "tumor_ratio": round(tumor_ratio, 4),
        "has_tumor": bool(tumor_pixels > 0),
        "category": category,
    }


def extract_slices_from_nifti(
    volume_path: str,
    seg_path: str,
    output_dir: str,
    subject_id: str,
    min_liver_pixels: int = 100,
    target_shape: Tuple[int, int] = TARGET_SIZE
) -> List[Dict[str, Any]]:
    """
    Extracts informative 2D axial slices from 3D NIfTI volume and segmentation.
    Discards non-liver axial slices.
    """
    try:
        import nibabel as nib
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError("nibabel and scipy are required for NIfTI processing. Install via requirements.txt.")

    vol_nii = nib.load(volume_path)
    seg_nii = nib.load(seg_path)

    vol_data = vol_nii.get_fdata()
    seg_data = seg_nii.get_fdata().astype(np.uint8)

    os.makedirs(output_dir, exist_ok=True)
    extracted_records = []

    # Axial slices along the Z-axis (axis 2)
    num_slices = vol_data.shape[2]

    for z in range(num_slices):
        img_slice = vol_data[:, :, z]
        mask_slice = seg_data[:, :, z]

        meta = compute_slice_metadata(mask_slice)
        # Skip slices without liver tissue
        if meta["liver_pixels"] < min_liver_pixels:
            continue

        # Window and normalize HU
        img_norm = apply_hu_window(img_slice)

        # Resize if necessary
        if img_norm.shape != target_shape:
            scale_y = target_shape[0] / img_norm.shape[0]
            scale_x = target_shape[1] / img_norm.shape[1]
            img_norm = zoom(img_norm, (scale_y, scale_x), order=1)
            mask_slice = zoom(mask_slice, (scale_y, scale_x), order=0)

        filename = f"{subject_id}_slice_{z:04d}.npz"
        filepath = os.path.join(output_dir, filename)

        np.savez_compressed(
            filepath,
            image=img_norm.astype(np.float32),
            mask=mask_slice.astype(np.uint8)
        )

        record = {
            "file": filename,
            "path": filepath,
            "subject_id": subject_id,
            "slice_index": z,
            **meta
        }
        extracted_records.append(record)

    return extracted_records
