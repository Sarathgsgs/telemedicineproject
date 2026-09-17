"""
tests/test_privacy.py
---------------------
Automated unit tests for Phase 6:
- L2 gradient/parameter clipping to threshold C
- Calibrated Gaussian noise calculation & injection
- Perfect cancellation of Zero-Sum Additive Masking (sum_i w_masked_i == sum_i w_i)
- Privacy accountant composition tracking
"""

import unittest
import numpy as np

from privacy.dp_mechanism import (
    clip_parameter_updates,
    compute_gaussian_sigma,
    apply_differential_privacy,
    PrivacyAccountant
)
from privacy.additive_masking import AdditiveMaskingManager


class TestPrivacyMechanisms(unittest.TestCase):

    def test_l2_norm_clipping(self):
        """Verifies that vector norms exceeding C are scaled strictly to C."""
        # Vector with norm = 5.0
        params = [np.array([3.0, 4.0], dtype=np.float32)]
        clipped, original_norm = clip_parameter_updates(params, clip_norm=1.0)

        self.assertAlmostEqual(original_norm, 5.0, places=4)
        new_norm = np.sqrt(np.sum(np.square(clipped[0])))
        self.assertAlmostEqual(float(new_norm), 1.0, places=4)

        # Vector with norm = 0.5 (already <= 1.0)
        small_params = [np.array([0.3, 0.4], dtype=np.float32)]
        unmodified, orig_small = clip_parameter_updates(small_params, clip_norm=1.0)
        self.assertAlmostEqual(orig_small, 0.5, places=4)
        self.assertTrue(np.allclose(unmodified[0], small_params[0]))

    def test_gaussian_sigma_scaling(self):
        """Verifies sigma scales inversely with epsilon."""
        sigma_tight = compute_gaussian_sigma(epsilon=0.1, delta=1e-5, clip_norm=1.0)
        sigma_relaxed = compute_gaussian_sigma(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        sigma_inf = compute_gaussian_sigma(epsilon=float("inf"), delta=1e-5, clip_norm=1.0)

        self.assertGreater(sigma_tight, sigma_relaxed)
        self.assertAlmostEqual(sigma_tight / sigma_relaxed, 10.0, places=2)
        self.assertEqual(sigma_inf, 0.0)

    def test_differential_privacy_injection(self):
        """Verifies noisy parameters differ from raw parameters when epsilon is finite."""
        raw_params = [np.zeros((10, 10), dtype=np.float32)]
        noisy, meta = apply_differential_privacy(raw_params, epsilon=1.0, delta=1e-5, seed=42)

        self.assertGreater(meta["sigma"], 0.0)
        # Check that noise was actually added
        diff = np.max(np.abs(noisy[0] - raw_params[0]))
        self.assertGreater(float(diff), 0.0)

    def test_zero_sum_additive_masking_perfect_cancellation(self):
        """
        Critical cryptographic test:
        1. Individual masked updates appear completely obscured.
        2. Server summation cancels all random masks to EXACTLY zero: sum_i w_masked = sum_i w.
        """
        num_clients = 3
        layer_shapes = [(5, 5), (10,)]
        
        mgr = AdditiveMaskingManager(num_clients=num_clients, layer_shapes=layer_shapes, seed=99)

        # 3 clients with distinct parameter weights
        client_weights = [
            [np.full((5, 5), 1.0, dtype=np.float32), np.full((10,), 2.0, dtype=np.float32)],
            [np.full((5, 5), 3.0, dtype=np.float32), np.full((10,), 4.0, dtype=np.float32)],
            [np.full((5, 5), 5.0, dtype=np.float32), np.full((10,), 6.0, dtype=np.float32)],
        ]

        # Calculate true unmasked sum
        true_sum_layer0 = client_weights[0][0] + client_weights[1][0] + client_weights[2][0]  # All 9.0
        true_sum_layer1 = client_weights[0][1] + client_weights[1][1] + client_weights[2][1]  # All 12.0

        # Mask each client
        masked_updates = []
        for cid in range(num_clients):
            m_params = mgr.mask_client_parameters(cid, client_weights[cid])
            masked_updates.append(m_params)
            # Verify individual weights are completely obscured
            self.assertFalse(np.allclose(m_params[0], client_weights[cid][0]))

        # Server aggregates
        unmasked_result = mgr.aggregate_masked_parameters(masked_updates)

        # Perfect cancellation test:
        self.assertTrue(np.allclose(unmasked_result[0], true_sum_layer0, atol=1e-6))
        self.assertTrue(np.allclose(unmasked_result[1], true_sum_layer1, atol=1e-6))

    def test_privacy_accountant(self):
        """Verifies analytical privacy accounting accumulation."""
        accountant = PrivacyAccountant(target_delta=1e-5)
        accountant.step(sigma=2.5)
        accountant.step(sigma=2.5)
        res = accountant.get_cumulative_privacy(clip_norm=1.0)
        self.assertEqual(res["rounds"], 2)
        self.assertIsInstance(res["cumulative_epsilon"], float)


if __name__ == "__main__":
    unittest.main()
