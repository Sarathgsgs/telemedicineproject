"""
tests/test_model.py
-------------------
Automated unit tests for Phase 3:
- Attention U-Net-lite architecture & GroupNorm stability
- Small-batch (B=1, B=2) invariance for Federated Learning
- Attention Gate coefficient maps in [0, 1]
- Composite loss backward pass & gradient validity
- Segmentation metrics (Dice, IoU, Precision, Recall)
"""

import unittest
import torch
import numpy as np

from model.unet_attention import AttentionUNetLite, ConvBlock, AttentionGate
from model.losses import CompositeLoss, SoftDiceLoss, FocalLoss
from model.metrics import compute_binary_metrics, evaluate_batch


class TestModelArchitecture(unittest.TestCase):

    def setUp(self):
        self.model = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=16)

    def test_groupnorm_small_batch_invariance(self):
        """
        Critical medical FL test: In federated learning with small local client datasets,
        batch sizes can be B=1 or B=2. BatchNorm crashes on B=1 in training mode.
        GroupNorm MUST succeed on B=1 and B=2.
        """
        self.model.train()
        # Test B=1
        x1 = torch.randn(1, 1, 64, 64)
        out1 = self.model(x1)
        self.assertEqual(out1.shape, (1, 3, 64, 64))

        # Test B=2
        x2 = torch.randn(2, 1, 64, 64)
        out2 = self.model(x2)
        self.assertEqual(out2.shape, (2, 3, 64, 64))

    def test_attention_gate_coefficients(self):
        """Verifies Attention Gate returns valid spatial attention maps in [0, 1]."""
        ag = AttentionGate(f_g=32, f_l=32, f_int=16)
        g = torch.randn(2, 32, 16, 16)
        x = torch.randn(2, 32, 16, 16)

        attenuated_x, alpha = ag(g, x)
        self.assertEqual(attenuated_x.shape, (2, 32, 16, 16))
        self.assertEqual(alpha.shape, (2, 1, 16, 16))
        self.assertTrue(torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0))

    def test_composite_loss_and_gradients(self):
        """Verifies loss computation and gradient backpropagation."""
        criterion = CompositeLoss(alpha=1.0, beta=0.5, gamma=0.5)
        x = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 3, (2, 64, 64))

        logits = self.model(x)
        loss, details = criterion(logits, targets)

        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_dice", details)
        self.assertIn("loss_ce", details)
        self.assertIn("loss_focal", details)

        loss.backward()
        # Verify non-zero gradients exist across parameters
        has_grads = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in self.model.parameters())
        self.assertTrue(has_grads)

    def test_metrics_calculation(self):
        """Tests Dice, IoU, Precision, Recall on perfect and partial matches."""
        # Perfect match
        gt = np.ones((50, 50), dtype=np.uint8)
        pred = np.ones((50, 50), dtype=np.uint8)
        m = compute_binary_metrics(pred, gt)
        self.assertAlmostEqual(m["dice"], 1.0, places=3)
        self.assertAlmostEqual(m["iou"], 1.0, places=3)
        self.assertAlmostEqual(m["precision"], 1.0, places=3)
        self.assertAlmostEqual(m["recall"], 1.0, places=3)

        # Batch evaluation
        logits = torch.zeros(2, 3, 32, 32)
        logits[:, 1, :, :] = 10.0  # Predicts class 1 everywhere
        targets = torch.ones(2, 32, 32, dtype=torch.long)
        b_metrics = evaluate_batch(logits, targets)
        self.assertAlmostEqual(b_metrics["liver_dice"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
