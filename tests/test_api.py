"""
tests/test_api.py
-----------------
Automated test suite for Phase 9:
- Teleconsultation API endpoints (/api/status, /api/samples, /api/predict_sample).
- Custom upload inference (/api/predict_upload).
- Offline mode toggling and sync replay (/api/toggle_offline, /api/sync_offline).
- Federated Learning Observatory data (/api/fl_observatory).
- Clinician feedback logging (/api/feedback).
"""

import io
import unittest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.main import app, sync_manager


class TestTelemedicineAPI(unittest.TestCase):
    """Verifies FastAPI endpoints for teleconsultation, inference, and sync."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertIn("model_architecture", data)
        self.assertIn("offline_queue", data)

    def test_samples_endpoint(self):
        res = self.client.get("/api/samples")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("samples", data)
        self.assertGreaterEqual(len(data["samples"]), 4)

    def test_sample_details_and_prediction(self):
        # 1. Test sample details
        res = self.client.get("/api/sample/slice_0001")
        self.assertEqual(res.status_code, 200)
        details = res.json()
        self.assertIn("raw_slice_b64", details)
        self.assertIn("gt_mask_b64", details)

        # 2. Test predict_sample
        res_pred = self.client.post("/api/predict_sample", json={"sample_id": "slice_0001"})
        self.assertEqual(res_pred.status_code, 200)
        pred = res_pred.json()
        self.assertIn("prediction", pred)
        self.assertIn("colored_mask_b64", pred)
        self.assertIn("tumor_area_pct", pred["prediction"])
        self.assertIn("mean_confidence", pred["prediction"])
        self.assertGreater(pred["prediction"]["inference_ms"], 0.0)

    def test_predict_upload(self):
        # Create a synthetic 128x128 test image
        img_arr = (np.random.rand(128, 128) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        res = self.client.post(
            "/api/predict_upload",
            files={"file": ("test_slice.png", buf, "image/png")},
            data={"patient_id": "PATIENT_TEST_UPLOAD"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("prediction", data)
        self.assertEqual(data["prediction"]["patient_id"], "PATIENT_TEST_UPLOAD")

    def test_offline_toggle_and_sync(self):
        # 1. Switch to offline
        res_off = self.client.post("/api/toggle_offline", json={"online": False})
        self.assertEqual(res_off.status_code, 200)
        self.assertEqual(res_off.json()["network_status"], "OFFLINE")

        # 2. Queue simulated update
        res_q = self.client.post("/api/queue_simulation_update")
        self.assertEqual(res_q.status_code, 200)
        self.assertTrue(res_q.json()["success"])

        # 3. Attempt sync while offline -> must defer
        res_sync_off = self.client.post("/api/sync_offline")
        self.assertEqual(res_sync_off.status_code, 200)
        self.assertFalse(res_sync_off.json()["sync_summary"]["sync_performed"])

        # 4. Switch back to online and sync
        self.client.post("/api/toggle_offline", json={"online": True})
        res_sync_on = self.client.post("/api/sync_offline")
        self.assertEqual(res_sync_on.status_code, 200)
        self.assertTrue(res_sync_on.json()["sync_summary"]["sync_performed"])

    def test_fl_observatory_endpoint(self):
        res = self.client.get("/api/fl_observatory")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("hospitals", data)
        self.assertEqual(len(data["hospitals"]), 3)
        self.assertIn("table4_personalization", data)
        self.assertIn("table7_communication", data)

    def test_feedback_endpoint(self):
        payload = {
            "patient_id": "PAT_TEST",
            "slice_id": "slice_0001",
            "clinician_id": "DR_UNIT_TEST",
            "verified_diagnosis": "Focal HCC Approved",
            "clinician_notes": "Segmentations concordant with contrast CT."
        }
        res = self.client.post("/api/feedback", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertIn("feedback_id", res.json())


if __name__ == "__main__":
    unittest.main()
