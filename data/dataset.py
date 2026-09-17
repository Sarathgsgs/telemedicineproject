"""
data/dataset.py
---------------
PyTorch Dataset and DataLoader constructors for liver CT slice segmentation.
Includes spatial data augmentations (rotation, scaling, flips, Gaussian noise).
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
from scipy.ndimage import rotate, zoom


class LiverSliceDataset(Dataset):
    """
    PyTorch Dataset for 2D axial abdominal CT slices.
    Loads pre-extracted/synthesized .npz files containing:
      - 'image': float32 array in [0, 1]
      - 'mask': uint8 array with values in {0, 1, 2}
    """
    def __init__(
        self,
        records: List[Dict[str, Any]],
        augment: bool = False,
        target_size: Tuple[int, int] = (256, 256)
    ):
        self.records = records
        self.augment = augment
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.records)

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies paired spatial augmentations to CT image and segmentation mask.
        """
        # 1. Random Horizontal / Vertical Flips
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        # 2. Random Rotation (-15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15.0, 15.0)
            image = rotate(image, angle, reshape=False, order=1, mode='nearest')
            mask = rotate(mask, angle, reshape=False, order=0, mode='nearest')

        # 3. Random Scaling (0.9 to 1.1x zoom)
        if random.random() > 0.5:
            factor = random.uniform(0.92, 1.08)
            h, w = image.shape
            
            # Zoom and crop/pad back to original shape
            zoomed_img = zoom(image, factor, order=1)
            zoomed_msk = zoom(mask, factor, order=0)
            
            zh, zw = zoomed_img.shape
            if factor > 1.0:
                # Crop center
                start_h = (zh - h) // 2
                start_w = (zw - w) // 2
                image = zoomed_img[start_h : start_h + h, start_w : start_w + w]
                mask = zoomed_msk[start_h : start_h + h, start_w : start_w + w]
            else:
                # Pad to original size
                pad_h = (h - zh) // 2
                pad_w = (w - zw) // 2
                new_img = np.zeros((h, w), dtype=np.float32)
                new_msk = np.zeros((h, w), dtype=np.uint8)
                new_img[pad_h : pad_h + zh, pad_w : pad_w + zw] = zoomed_img
                new_msk[pad_h : pad_h + zh, pad_w : pad_w + zw] = zoomed_msk
                image = new_img
                mask = new_msk

        # 4. Subtle Gaussian Noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.015, size=image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.records[idx]
        filepath = record["filepath"]

        # Load compressed archive
        with np.load(filepath) as data:
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.int64)

        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # PyTorch format: image is (1, H, W), mask is (H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor, record


def get_dataloader(
    records: List[Dict[str, Any]],
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Constructs a PyTorch DataLoader for training or evaluation.
    """
    dataset = LiverSliceDataset(records, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )
