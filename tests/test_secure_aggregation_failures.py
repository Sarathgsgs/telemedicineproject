"""
tests/test_secure_aggregation_failures.py
-----------------------------------------
Automated test suite for Phase 3:
- Verifies mathematical cancellation of Zero-Sum Additive Masking under full participation.
- Verifies safe detection of client dropouts (mask imbalance prevents corrupted aggregation).
- Verifies server blindness: individual masked payloads conceal unmasked hospital updates.
- Verifies PrivacyAccountant cumulative (epsilon, delta) budget tracking over successive rounds.
"""

import unittest
import numpy as np

from privacy.additive_masking import AdditiveMaskingManager
from privacy.dp_mechanism import PrivacyAccountant, compute_gaussian_sigma, apply_differential_privacy


class TestSecureAggregationAndPrivacy(unittest.TestCase):
    """Rigorous cryptographic tests for Additive Masking and DP-SGD."""

    def setUp(self):
        self.num_clients = 3
        self.layer_shapes = [(32, 1, 3, 3), (32,), (64, 32, 3, 3), (64,)]
        self.mask_mgr = AdditiveMaskingManager(
            num_clients=self.num_clients,
            layer_shapes=self.layer_shapes,
            seed=101
        )

        # Generate mock weight updates for 3 clients
        self.client_weights = []
        for c in range(self.num_clients):
            c_w = [np.random.normal(c * 0.1, 0.05, size=s).astype(np.float32) for s in self.layer_shapes]
            self.client_weights.append(c_w)

    def test_full_participation_perfect_cancellation(self):
        """Verifies that all pairwise masks sum to exactly 0 when all clients participate."""
        masked_payloads = []
        for cid in range(self.num_clients):
            masked = self.mask_mgr.mask_client_parameters(cid, self.client_weights[cid])
            masked_payloads.append(masked)

        # Server sums masked updates
        aggregated_sum = self.mask_mgr.aggregate_masked_parameters(masked_payloads)

        # Ground truth unmasked sum
        expected_sum = []
        for l_idx in range(len(self.layer_shapes)):
            layer_sum = sum(self.client_weights[c][l_idx] for c in range(self.num_clients))
            expected_sum.append(layer_sum)

        for l_idx in range(len(self.layer_shapes)):
            max_diff = np.max(np.abs(aggregated_sum[l_idx] - expected_sum[l_idx]))
            self.assertLess(max_diff, 1e-5, f"Mask cancellation failed on layer {l_idx} with error {max_diff}")

    def test_client_dropout_triggers_mask_imbalance(self):
        """Verifies that if a client drops offline, masks do not sum to zero (preventing silent corruption)."""
        masked_payloads = []
        for cid in range(self.num_clients):
            masked = self.mask_mgr.mask_client_parameters(cid, self.client_weights[cid])
            masked_payloads.append(masked)

        # Simulate Client 2 dropping offline mid-round (only clients 0 and 1 send payloads)
        dropped_client = 2
        partial_payloads = [masked_payloads[0], masked_payloads[1]]

        corrupted_sum = self.mask_mgr.aggregate_masked_parameters(partial_payloads)
        expected_two_client_sum = [
            self.client_weights[0][l] + self.client_weights[1][l]
            for l in range(len(self.layer_shapes))
        ]

        # The difference must be NON-ZERO because M_{0,2} and M_{1,2} are uncancelled
        has_imbalance = False
        for l_idx in range(len(self.layer_shapes)):
            diff = np.max(np.abs(corrupted_sum[l_idx] - expected_two_client_sum[l_idx]))
            if diff > 0.1:
                has_imbalance = True
                break

        self.assertTrue(
            has_imbalance,
            "Safe failure check failed: dropped client masks must create an imbalance alerting the server."
        )

    def test_server_blindness_individual_weights(self):
        """Verifies that an individual masked update vector conceals the true hospital weights."""
        cid = 0
        original = self.client_weights[cid]
        masked = self.mask_mgr.mask_client_parameters(cid, original)

        # Confirm masked vector has high L2 deviation from raw vector
        total_deviation = 0.0
        for l_idx in range(len(self.layer_shapes)):
            total_deviation += float(np.linalg.norm(masked[l_idx] - original[l_idx]))

        self.assertGreater(
            total_deviation, 1.0,
            "Masking was insufficient; individual payload remains too close to raw weights."
        )

    def test_privacy_accountant_composition(self):
        """Verifies PrivacyAccountant tracks cumulative epsilon growth over successive rounds."""
        accountant = PrivacyAccountant(target_delta=1e-5)
        sigma = compute_gaussian_sigma(epsilon=1.0, delta=1e-5, clip_norm=1.0)

        # Round 1
        accountant.step(sigma)
        res_r1 = accountant.get_cumulative_privacy(clip_norm=1.0)
        eps_r1 = res_r1["cumulative_epsilon"]

        # Round 2
        accountant.step(sigma)
        res_r2 = accountant.get_cumulative_privacy(clip_norm=1.0)
        eps_r2 = res_r2["cumulative_epsilon"]

        # Round 3
        accountant.step(sigma)
        res_r3 = accountant.get_cumulative_privacy(clip_norm=1.0)
        eps_r3 = res_r3["cumulative_epsilon"]

        self.assertEqual(res_r3["rounds"], 3)
        self.assertGreater(eps_r3, eps_r1, "Cumulative epsilon must increase with additional rounds.")


if __name__ == "__main__":
    unittest.main()
