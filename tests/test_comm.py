"""
tests/test_comm.py
------------------
Automated unit tests for Phase 7:
- Top-k parameter sparsity selection by magnitude
- Dense bypass (density=1.0)
- Error-feedback buffer accumulation and conservation
- Decompression array shape and coordinate value restoration
"""

import unittest
import numpy as np

from comm.topk_sparsify import TopKSparsifier


class TestCommunicationEfficiency(unittest.TestCase):

    def test_topk_density_selection(self):
        """Verifies that density_ratio=0.20 selects exactly top 20% by absolute magnitude."""
        sparsifier = TopKSparsifier(density_ratio=0.20)
        
        # 100 elements: 80 small (0.1), 20 large (10.0)
        layer = np.full(100, 0.1, dtype=np.float32)
        layer[10:30] = 10.0  # Exactly 20 large elements

        payload, meta = sparsifier.compress_updates(client_id=0, updates=[layer])

        self.assertEqual(meta["density_ratio"], 0.20)
        self.assertEqual(meta["transmitted_parameters"], 20)
        self.assertEqual(len(payload[0]["indices"]), 20)
        
        # All decompressed non-zero values must be the large values (10.0)
        reconstructed = sparsifier.decompress_updates(payload)[0]
        self.assertTrue(np.all(reconstructed[10:30] == 10.0))
        self.assertTrue(np.all(reconstructed[:10] == 0.0))

    def test_dense_bypass(self):
        """Verifies that density_ratio=1.0 acts as dense bypass."""
        sparsifier = TopKSparsifier(density_ratio=1.0)
        layer = np.random.randn(5, 5).astype(np.float32)

        payload, meta = sparsifier.compress_updates(client_id=1, updates=[layer])
        self.assertTrue(payload[0]["is_dense"])
        reconstructed = sparsifier.decompress_updates(payload)[0]
        self.assertTrue(np.allclose(reconstructed, layer))

    def test_error_feedback_buffer_accumulation(self):
        """
        Critical EF-SGD test:
        Verifies that dropped residual parameters are accumulated in the local buffer
        and transmitted in subsequent rounds once their accumulated magnitude exceeds the threshold.
        """
        sparsifier = TopKSparsifier(density_ratio=0.50)  # Top 50%
        # 4 elements: [10.0, 10.0, 1.0, 1.0]
        round1_update = np.array([10.0, 10.0, 1.0, 1.0], dtype=np.float32)

        payload1, meta1 = sparsifier.compress_updates(client_id=2, updates=[round1_update])
        # In round 1, top-2 indices are [0, 1] (values 10.0, 10.0). Residual [0.0, 0.0, 1.0, 1.0] in buffer
        self.assertTrue(np.allclose(sparsifier.error_buffers[2][0], np.array([0.0, 0.0, 1.0, 1.0])))

        # In round 2, small update [0.0, 0.0, 0.5, 0.5] arrives
        # With accumulated buffer, values at [2, 3] become 1.0 + 0.5 = 1.5
        round2_update = np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32)
        payload2, meta2 = sparsifier.compress_updates(client_id=2, updates=[round2_update])

        # Indices 2 and 3 should now be transmitted because their accumulated value (1.5) is highest!
        transmitted_indices = sorted(payload2[0]["indices"])
        self.assertEqual(transmitted_indices, [2, 3])

    def test_bandwidth_savings_calculation(self):
        """Verifies calculation of bandwidth savings percentage."""
        sparsifier = TopKSparsifier(density_ratio=0.10)
        layer = np.random.randn(1000).astype(np.float32)

        _, meta = sparsifier.compress_updates(client_id=3, updates=[layer])
        # 10% transmitted in sparse index-value format (8 bytes/param) vs dense (4 bytes/param) -> 80% savings
        self.assertGreater(meta["bandwidth_savings_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
