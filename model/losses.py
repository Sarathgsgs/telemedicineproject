"""
model/losses.py
---------------
Composite loss function matching FedLiverNet Eq. 4:
L_total = alpha * L_Dice + beta * L_CE + gamma * L_Focal
Specially weighted to resolve the extreme foreground/background class imbalance
between vast abdominal background tissue, liver parenchyma, and small focal tumors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss with Laplace smoothing.
    """
    def __init__(self, smooth: float = 1e-5, ignore_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes, H, W)
            targets: (B, H, W) integer class indices
        """
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        # Convert targets to one-hot: (B, num_classes, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        start_c = 1 if self.ignore_background else 0
        dice_total = 0.0
        active_classes = 0

        for c in range(start_c, num_classes):
            p_c = probs[:, c].contiguous().view(-1)
            t_c = targets_one_hot[:, c].contiguous().view(-1)

            intersection = (p_c * t_c).sum()
            cardinality = (p_c * p_c).sum() + (t_c * t_c).sum()

            dice_c = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_total += dice_c
            active_classes += 1

        return 1.0 - (dice_total / max(active_classes, 1))


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., ICCV 2017).
    Downweights well-classified easy background pixels and focuses learning
    on ambiguous liver capsule and focal tumor boundary voxels.
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes, H, W)
            targets: (B, H, W)
        """
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        # Gather probability of target class
        targets_expanded = targets.unsqueeze(1)
        log_pt = log_p.gather(1, targets_expanded).squeeze(1)
        pt = p.gather(1, targets_expanded).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha.gather(0, targets.view(-1)).view(targets.shape)
            focal_loss = -alpha_t * focal_weight * log_pt
        else:
            focal_loss = -focal_weight * log_pt

        return focal_loss.mean()


class CompositeLoss(nn.Module):
    """
    Composite Loss for Liver & Tumor CT Segmentation:
    L = alpha * L_Dice + beta * L_CE + gamma * L_Focal
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Default class weights: penalizes missing tumor (Class 2) 4x more than background
        if class_weights is None:
            # 0: Background (0.2), 1: Liver (1.0), 2: Tumor (3.5)
            class_weights = torch.tensor([0.2, 1.0, 3.5], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)

        self.dice_loss = SoftDiceLoss(ignore_background=False)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.focal_loss = FocalLoss(gamma=2.0, alpha=self.class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_dice = self.dice_loss(logits, targets)
        l_ce = self.ce_loss(logits, targets)
        l_focal = self.focal_loss(logits, targets)

        total_loss = self.alpha * l_dice + self.beta * l_ce + self.gamma * l_focal
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_dice": l_dice.item(),
            "loss_ce": l_ce.item(),
            "loss_focal": l_focal.item()
        }
