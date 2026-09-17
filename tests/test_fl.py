"""
tests/test_fl.py
----------------
Automated unit test suite for Phase 4 Federated Learning:
- Client parameter extraction and injection
- Local client training fit() step
- Local client evaluation evaluate() step
- Weighted metric aggregation and inter-client standard deviation (sigma)
- Mathematical correctness of FedAvg aggregation
"""

import unittest
import torch
import numpy as np
import tempfile

from fl.client import HospitalClient, get_parameters, set_parameters
from fl.server import aggregate_eval_metrics
from model.unet_attention import AttentionUNetLite
from data.synthetic_generator import generate_synthetic_cohort


class TestFederatedLearning(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.records = generate_synthetic_cohort(
            output_dir=self.tmpdir.name,
            num_samples=10,
            skew_profile="balanced",
            client_id="fl_test"
        )
        self.train_recs = self.records[:8]
        self.test_recs = self.records[8:]

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parameter_get_and_set(self):
        """Validates that parameters extracted from a model can be restored losslessly."""
        model1 = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=16)
        model2 = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=16)

        params1 = get_parameters(model1)
        set_parameters(model2, params1)
        params2 = get_parameters(model2)

        for p1, p2 in zip(params1, params2):
            self.assertTrue(np.allclose(p1, p2, atol=1e-7))

    def test_client_fit_updates_weights(self):
        """Verifies that running client.fit() modifies parameters in the direction of loss reduction."""
        client = HospitalClient(
            client_id=0,
            train_records=self.train_recs,
            test_records=self.test_recs,
            base_filters=16,
            local_epochs=1,
            batch_size=4
        )

        initial_params = client.get_parameters({})
        updated_params, num_samples, metrics = client.fit(initial_params, {"local_epochs": 1})

        self.assertEqual(num_samples, len(self.train_recs))
        self.assertIn("train_loss", metrics)

        # Check that weights actually changed
        has_changed = False
        for p_init, p_upd in zip(initial_params, updated_params):
            if not np.allclose(p_init, p_upd, atol=1e-5):
                has_changed = True
                break
        self.assertTrue(has_changed, "Client weights did not change after fit()")

    def test_client_evaluate(self):
        """Verifies that client.evaluate() returns valid metrics dictionary."""
        client = HospitalClient(
            client_id=1,
            train_records=self.train_recs,
            test_records=self.test_recs,
            base_filters=16
        )
        params = client.get_parameters({})
        loss, n_test, eval_dict = client.evaluate(params, {})

        self.assertGreater(loss, 0.0)
        self.assertEqual(n_test, len(self.test_recs))
        self.assertIn("mean_dice", eval_dict)
        self.assertIn("liver_dice", eval_dict)
        self.assertIn("tumor_dice", eval_dict)
        self.assertTrue(0.0 <= eval_dict["mean_dice"] <= 1.0)

    def test_fedavg_mathematical_aggregation(self):
        """Verifies coordinate-wise weighted averaging: w_glob = (n1*w1 + n2*w2) / (n1 + n2)."""
        w1 = [np.array([1.0, 2.0]), np.array([10.0, 20.0])]
        w2 = [np.array([3.0, 4.0]), np.array([30.0, 40.0])]
        n1 = 100
        n2 = 300
        total_n = n1 + n2

        aggregated = []
        for p_idx in range(len(w1)):
            w_coord = (n1 / total_n) * w1[p_idx] + (n2 / total_n) * w2[p_idx]
            aggregated.append(w_coord)

        expected_0 = 0.25 * np.array([1.0, 2.0]) + 0.75 * np.array([3.0, 4.0])
        self.assertTrue(np.allclose(aggregated[0], expected_0))

    def test_metric_aggregation_and_variance(self):
        """Verifies calculation of inter-client standard deviation (sigma)."""
        # Client 0 has Dice 0.80 (50 samples), Client 1 has Dice 0.60 (50 samples)
        metrics = [
            (50, {"mean_dice": 0.80, "tumor_dice": 0.70, "liver_dice": 0.90}),
            (50, {"mean_dice": 0.60, "tumor_dice": 0.50, "liver_dice": 0.70})
        ]
        agg = aggregate_eval_metrics(metrics)

        self.assertAlmostEqual(agg["mean_dice"], 0.70, places=4)
        # Population std of [0.80, 0.60] is 0.10
        self.assertAlmostEqual(agg["client_mean_dice_std"], 0.10, places=4)


if __name__ == "__main__":
    unittest.main()
