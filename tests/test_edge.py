"""
tests/test_edge.py
------------------
Automated test suite for Phase 8:
- TorchScript export verification and eager model parity.
- EdgeInferenceEngine diagnostic prediction and HU windowing.
- OfflineSyncManager SQLite transactional queueing and sync replay.
- Network disconnection handling and error tolerance.
"""

import os
import unittest
import tempfile
import numpy as np
import torch

from model.unet_attention import AttentionUNetLite
from model.export_edge import export_to_torchscript, benchmark_edge_latency
from edge.offline_manager import EdgeInferenceEngine, OfflineSyncManager, EdgePredictionResult


class TestEdgeExport(unittest.TestCase):
    """Verifies TorchScript tracing and numerical parity."""

    def setUp(self):
        self.model = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=16)
        self.model.eval()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.export_path = os.path.join(self.temp_dir.name, "test_edge_model.pt")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_and_numerical_parity(self):
        traced_model, metrics = export_to_torchscript(
            self.model,
            self.export_path,
            input_shape=(1, 1, 64, 64),
            device="cpu"
        )
        self.assertTrue(metrics["numerical_parity"])
        self.assertLess(metrics["max_abs_difference"], 1e-4)
        self.assertTrue(os.path.exists(self.export_path))

    def test_edge_latency_benchmark(self):
        traced_model, _ = export_to_torchscript(
            self.model,
            self.export_path,
            input_shape=(1, 1, 64, 64),
            device="cpu"
        )
        bench = benchmark_edge_latency(
            traced_model,
            input_shape=(1, 1, 64, 64),
            num_warmup=2,
            num_iterations=5,
            device="cpu"
        )
        self.assertIn("mean_latency_ms", bench)
        self.assertGreater(bench["throughput_slices_per_sec"], 0.0)


class TestEdgeInferenceEngine(unittest.TestCase):
    """Verifies EdgeInferenceEngine preprocessing and prediction."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "engine_model.pt")
        m = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=16)
        m.eval()
        torch.save(m.state_dict(), self.model_path)
        self.engine = EdgeInferenceEngine(model_path=self.model_path, device="cpu", base_filters=16)

    def tearDown(self):
        if hasattr(self, "engine"):
            del self.engine
        import gc
        gc.collect()
        self.temp_dir.cleanup()

    def test_slice_preprocessing(self):
        # Raw HU slice with extreme values
        raw_slice = np.random.uniform(-1000, 1000, size=(128, 128)).astype(np.float32)
        tensor = self.engine.preprocess_slice(raw_slice, target_size=(128, 128))
        self.assertEqual(tensor.shape, (1, 1, 128, 128))
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_predict_structure(self):
        dummy_slice = np.random.uniform(0.0, 1.0, size=(128, 128)).astype(np.float32)
        result = self.engine.predict(dummy_slice, patient_id="TEST_001", slice_id="SL_10", is_offline=True)
        self.assertIsInstance(result, EdgePredictionResult)
        self.assertEqual(result.patient_id, "TEST_001")
        self.assertEqual(result.slice_id, "SL_10")
        self.assertTrue(result.is_offline)
        self.assertEqual(result.total_pixels, 128 * 128)
        self.assertEqual(result.segmentation_mask.shape, (128, 128))
        self.assertGreaterEqual(result.mean_confidence, 0.0)
        self.assertLessEqual(result.mean_confidence, 1.0)


class TestOfflineSyncManager(unittest.TestCase):
    """Verifies SQLite persistence, offline queueing, and reconnection sync."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_offline.db")
        self.manager = OfflineSyncManager(db_path=self.db_path)

    def tearDown(self):
        if hasattr(self, "manager"):
            self.manager.close()
            del self.manager
        import gc
        gc.collect()
        self.temp_dir.cleanup()

    def test_inference_audit_log(self):
        dummy_result = EdgePredictionResult(
            slice_id="SLICE_01",
            patient_id="PAT_01",
            liver_pixels=2500,
            tumor_pixels=120,
            total_pixels=16384,
            liver_area_pct=15.26,
            tumor_area_pct=0.73,
            mean_confidence=0.8842,
            tumor_detected=True,
            inference_ms=115.4,
            is_offline=True
        )
        row_id = self.manager.log_inference(dummy_result)
        self.assertGreater(row_id, 0)
        history = self.manager.get_recent_inferences(limit=5)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["patient_id"], "PAT_01")
        self.assertEqual(history[0]["is_offline"], 1)

    def test_queue_and_offline_deferral(self):
        # Ensure offline state
        self.manager.set_connectivity(False)

        # Queue two local updates
        sample_weights = [np.random.randn(5, 5).astype(np.float32) for _ in range(3)]
        id1 = self.manager.queue_fl_update(client_id=0, round_id=1, num_samples=100, update_data=sample_weights)
        id2 = self.manager.queue_fl_update(client_id=0, round_id=2, num_samples=120, update_data=sample_weights)

        self.assertGreater(id1, 0)
        self.assertGreater(id2, 0)

        status = self.manager.get_queue_status()
        self.assertEqual(status["network_status"], "OFFLINE")
        self.assertEqual(status["pending_fl_updates_count"], 2)
        self.assertGreater(status["pending_fl_updates_kb"], 0.0)

        # Attempt sync while offline -> must safely defer
        sync_res = self.manager.sync_all()
        self.assertFalse(sync_res["sync_performed"])
        self.assertEqual(sync_res["pending_fl_updates"], 2)

    def test_reconnection_and_sync_replay(self):
        # Queue updates offline
        self.manager.set_connectivity(False)
        dummy_update = {"dense_weights": [np.ones((2, 2))]}
        self.manager.queue_fl_update(client_id=1, round_id=1, num_samples=50, update_data=dummy_update)
        self.manager.queue_clinical_feedback(
            patient_id="PAT_02", slice_id="SL_02", clinician_id="DR_SMITH", verified_diagnosis="HCC Confirmed"
        )

        # Restore connectivity
        self.manager.set_connectivity(True)
        uploaded_payloads = []

        def mock_server_upload(payload):
            uploaded_payloads.append(payload)
            return True  # HTTP 200 ACK

        sync_summary = self.manager.sync_all(
            fl_server_upload_fn=mock_server_upload,
            feedback_upload_fn=mock_server_upload
        )

        self.assertTrue(sync_summary["sync_performed"])
        self.assertEqual(sync_summary["synced_fl_updates"], 1)
        self.assertEqual(sync_summary["synced_feedback"], 1)
        self.assertEqual(sync_summary["remaining_pending"], 0)
        self.assertEqual(len(uploaded_payloads), 2)

    def test_server_failure_retry_handling(self):
        self.manager.set_connectivity(True)
        self.manager.queue_fl_update(client_id=2, round_id=1, num_samples=80, update_data=[np.zeros((3, 3))])

        def failing_server(payload):
            return False  # Simulate server timeout or 500 error

        sync_summary = self.manager.sync_all(fl_server_upload_fn=failing_server)
        self.assertEqual(sync_summary["failed_fl_updates"], 1)
        self.assertEqual(sync_summary["synced_fl_updates"], 0)

        status = self.manager.get_queue_status()
        self.assertEqual(status["failed_fl_updates_count"], 1)


if __name__ == "__main__":
    unittest.main()
