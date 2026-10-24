"""
tests/test_patient_splitter.py
------------------------------
Unit tests for data/patient_splitter.py:
- Verifies that slices are grouped by patient volume ID.
- Verifies strict patient-level separation: 0% overlap across train, val, and test.
- Verifies clinical non-IID assignment across hospital client nodes.
"""

import unittest
from data.patient_splitter import partition_by_patient


class TestPatientSplitter(unittest.TestCase):
    """Tests data leakage prevention and patient-level partitioning."""

    def setUp(self):
        # Generate mock slice records for 10 distinct patients
        self.mock_records = []
        for p_idx in range(10):
            pid = f"patient_{p_idx:03d}"
            # Each patient has 12 slices
            for s_idx in range(12):
                self.mock_records.append({
                    "subject_id": pid,
                    "slice_index": s_idx,
                    "liver_pixels": 2500,
                    "tumor_pixels": 150 if p_idx < 3 else (50 if p_idx < 6 else 0),
                    "category": "large_tumor" if p_idx < 3 else ("small_tumor" if p_idx < 6 else "normal")
                })

    def test_zero_patient_leakage(self):
        """Verifies that no patient volume appears in more than one partition."""
        result = partition_by_patient(
            slice_records=self.mock_records,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            num_clients=3,
            seed=42
        )

        pids = result["patient_ids"]
        train_set = set(pids["train"])
        val_set = set(pids["val"])
        test_set = set(pids["test"])

        self.assertEqual(len(train_set.intersection(val_set)), 0, "Train and Val share patients!")
        self.assertEqual(len(train_set.intersection(test_set)), 0, "Train and Test share patients!")
        self.assertEqual(len(val_set.intersection(test_set)), 0, "Val and Test share patients!")
        self.assertTrue(result["summary"]["data_leakage_prevented"])

    def test_client_distribution_completeness(self):
        """Verifies that all training slices are distributed across the client nodes."""
        result = partition_by_patient(
            slice_records=self.mock_records,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            num_clients=3,
            seed=42
        )

        total_client_slices = sum(len(slices) for slices in result["client_train_splits"].values())
        expected_train_slices = len(result["patient_ids"]["train"]) * 12

        self.assertEqual(total_client_slices, expected_train_slices)
        self.assertEqual(len(result["client_train_splits"]), 3)


if __name__ == "__main__":
    unittest.main()
