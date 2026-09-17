"""
tests/test_data.py
------------------
Automated test suite for Phase 2 data pipeline:
- Clinical HU windowing & normalization
- Synthetic CT slice generation
- Controlled Non-IID client partitioning
- PyTorch Dataset & DataLoader extraction
"""

import unittest
import numpy as np
import torch
import tempfile
import os

from data.slice_extractor import apply_hu_window, compute_slice_metadata
from data.synthetic_generator import create_synthetic_ct_slice, generate_synthetic_cohort
from data.non_iid_splitter import partition_slices_non_iid, partition_slices_iid, summarize_partition
from data.dataset import LiverSliceDataset, get_dataloader


class TestDataPipeline(unittest.TestCase):

    def test_hu_windowing(self):
        """Validates HU clipping to [-100, 400] maps linearly to [0, 1]."""
        raw_hu = np.array([-1500.0, -100.0, 150.0, 400.0, 2000.0], dtype=np.float32)
        norm = apply_hu_window(raw_hu)

        self.assertEqual(norm[0], 0.0)  # Below -100 clipped to 0
        self.assertEqual(norm[1], 0.0)  # Exactly -100 is 0
        self.assertAlmostEqual(norm[2], 0.5, places=4)  # 150 is midpoint between -100 and 400
        self.assertEqual(norm[3], 1.0)  # Exactly 400 is 1.0
        self.assertEqual(norm[4], 1.0)  # Above 400 clipped to 1.0
        self.assertTrue(np.all((norm >= 0.0) & (norm <= 1.0)))

    def test_slice_metadata_calculation(self):
        """Checks voxel counting and lesion categorization."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1  # 1600 liver pixels
        mask[30:35, 30:35] = 2  # 25 tumor pixels (small)

        meta = compute_slice_metadata(mask)
        self.assertEqual(meta["liver_pixels"], 1600)
        self.assertEqual(meta["tumor_pixels"], 25)
        self.assertTrue(meta["has_tumor"])
        self.assertEqual(meta["category"], "small_tumor")

    def test_synthetic_slice_generation(self):
        """Checks slice synthesis shapes, data types, and values."""
        img, mask, meta = create_synthetic_ct_slice(size=128, tumor_type="large", seed=42)
        
        self.assertEqual(img.shape, (128, 128))
        self.assertEqual(mask.shape, (128, 128))
        self.assertEqual(img.dtype, np.float32)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all((img >= 0.0) & (img <= 1.0)))
        
        unique_labels = set(np.unique(mask))
        self.assertTrue(unique_labels.issubset({0, 1, 2}))
        self.assertIn(1, unique_labels)  # Liver should be present

    def test_controlled_non_iid_skew(self):
        """Verifies institutional clinical skew across 3 clients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records = generate_synthetic_cohort(
                output_dir=tmpdir,
                num_samples=60,
                skew_profile="balanced",
                client_id="test"
            )
            self.assertEqual(len(records), 60)

            splits = partition_slices_non_iid(records, num_clients=3, seed=42)
            summary = summarize_partition(splits)

            # Hospital 0: Metropolitan Cancer Center -> Biased toward large tumors
            h0_large = summary["client_0"]["category_breakdown"].get("large_tumor", 0)
            h1_large = summary["client_1"]["category_breakdown"].get("large_tumor", 0)
            self.assertGreaterEqual(h0_large, h1_large)

            # Hospital 2: Community Screening -> Biased toward normal scans
            h2_normal = summary["client_2"]["category_breakdown"].get("normal", 0)
            h0_normal = summary["client_0"]["category_breakdown"].get("normal", 0)
            self.assertGreaterEqual(h2_normal, h0_normal)

    def test_pytorch_dataset_and_loader(self):
        """Verifies PyTorch Dataset tensors and DataLoader batch shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records = generate_synthetic_cohort(
                output_dir=tmpdir,
                num_samples=16,
                skew_profile="balanced",
                client_id="test_loader"
            )

            dataset = LiverSliceDataset(records, augment=True)
            self.assertEqual(len(dataset), 16)

            img_t, mask_t, meta = dataset[0]
            self.assertEqual(img_t.shape, (1, 256, 256))
            self.assertEqual(mask_t.shape, (256, 256))
            self.assertEqual(img_t.dtype, torch.float32)
            self.assertEqual(mask_t.dtype, torch.int64)

            loader = get_dataloader(records, batch_size=4, shuffle=True, augment=True)
            for batch_img, batch_mask, _ in loader:
                self.assertEqual(batch_img.shape, (4, 1, 256, 256))
                self.assertEqual(batch_mask.shape, (4, 256, 256))
                break


if __name__ == "__main__":
    unittest.main()
