"""
tests/test_evaluation.py
------------------------
Automated test suite for Phase 10:
- Architectural ablations: GroupNorm vs BatchNorm, Attention Gates vs Vanilla U-Net.
- Loss formulation ablations: CompositeLoss, SoftDiceLoss.
- Master evaluation summary integrity and schema compliance.
"""

import os
import json
import unittest
import torch

from model.unet_ablations import UNetAblation, ConvBlockAblation
from model.losses import CompositeLoss, SoftDiceLoss


class TestAblationArchitectures(unittest.TestCase):
    """Verifies configurable U-Net variants for normalization and attention ablations."""

    def test_groupnorm_variant(self):
        model = UNetAblation(in_channels=1, num_classes=3, base_filters=16, norm_type="group_norm", use_attention=True)
        model.eval()
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 3, 64, 64))

    def test_batchnorm_variant(self):
        model = UNetAblation(in_channels=1, num_classes=3, base_filters=16, norm_type="batch_norm", use_attention=True)
        model.eval()
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 3, 64, 64))

    def test_vanilla_unet_no_attention(self):
        model = UNetAblation(in_channels=1, num_classes=3, base_filters=16, norm_type="group_norm", use_attention=False)
        model.eval()
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 3, 64, 64))


class TestLossAblations(unittest.TestCase):
    """Verifies loss function formulations."""

    def test_composite_loss_components(self):
        criterion = CompositeLoss(alpha=1.0, beta=0.5, gamma=0.5)
        logits = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, 3, (2, 32, 32))
        loss, components = criterion(logits, targets)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_dice", components)
        self.assertIn("loss_ce", components)
        self.assertIn("loss_focal", components)

    def test_soft_dice_loss(self):
        dice_criterion = SoftDiceLoss(ignore_background=False)
        logits = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, 3, (2, 32, 32))
        loss = dice_criterion(logits, targets)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0)


class TestMasterEvaluationSummary(unittest.TestCase):
    """Verifies master evaluation report existence and data schema."""

    def test_summary_json_schema(self):
        summary_path = "results/master_evaluation_summary.json"
        self.assertTrue(os.path.exists(summary_path), f"Summary file missing: {summary_path}")

        with open(summary_path, "r") as f:
            data = json.load(f)

        self.assertIn("table4_personalization", data)
        self.assertIn("table5_privacy_utility", data)
        self.assertIn("table6_ablations", data)
        self.assertIn("table7_communication", data)
        self.assertIn("table8_edge_profile", data)

        # Check Table 6 structure
        table6 = data["table6_ablations"]
        self.assertEqual(len(table6), 5)
        self.assertEqual(table6[0]["ablation_category"], "Complete Architecture (Ours)")

        # Verify plot exists
        plot_path = "results/master_evaluation_and_ablations.png"
        self.assertTrue(os.path.exists(plot_path), f"Plot file missing: {plot_path}")


if __name__ == "__main__":
    unittest.main()
