"""
model/metrics.py
----------------
Clinical segmentation metrics for liver parenchyma and focal tumor lesions:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard Index)
- Precision (Positive Predictive Value)
- Recall (Sensitivity)
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple


def compute_binary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6
) -> Dict[str, float]:
    """
    Computes binary Dice, IoU, Precision, Recall for a single class.
    """
    pred_b = (pred > 0).astype(np.bool_)
    target_b = (target > 0).astype(np.bool_)

    tp = np.logical_and(pred_b, target_b).sum()
    fp = np.logical_and(pred_b, ~target_b).sum()
    fn = np.logical_and(~pred_b, target_b).sum()

    if target_b.sum() == 0 and pred_b.sum() == 0:
        # True negative scan
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
    else:
        dice = float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))
        iou = float((tp + eps) / (tp + fp + fn + eps))
        precision = float((tp + eps) / (tp + fp + eps))
        recall = float((tp + eps) / (tp + fn + eps))

    return {
        "dice": float(np.clip(dice, 0.0, 1.0)),
        "iou": float(np.clip(iou, 0.0, 1.0)),
        "precision": float(np.clip(precision, 0.0, 1.0)),
        "recall": float(np.clip(recall, 0.0, 1.0))
    }


from scipy.ndimage import binary_erosion, distance_transform_edt


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float] = (1.0, 1.0),
    max_penalty: float = 100.0
) -> float:
    """
    Computes 95th percentile Hausdorff Distance (HD95) between binary masks in mm.
    
    Args:
        pred: Binary prediction mask.
        target: Binary ground-truth reference mask.
        voxel_spacing: In-plane pixel spacing in millimeters (default: (1.0, 1.0)).
        max_penalty: Penalty assigned when one mask is completely empty (default: 100.0 mm).
        
    Returns:
        HD95 metric in millimeters (lower is better, 0.0 is perfect contour match).
    """
    pred_b = (pred > 0).astype(bool)
    target_b = (target > 0).astype(bool)

    if not np.any(pred_b) and not np.any(target_b):
        return 0.0
    if not np.any(pred_b) or not np.any(target_b):
        return float(max_penalty)

    # Extract 1-pixel contour boundaries using binary erosion
    border_pred = pred_b ^ binary_erosion(pred_b)
    border_target = target_b ^ binary_erosion(target_b)

    if not np.any(border_pred) or not np.any(border_target):
        return float(max_penalty)

    # Compute Euclidean distance transforms
    dt_target = distance_transform_edt(~border_target, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~border_pred, sampling=voxel_spacing)

    # Distance from prediction boundary to target boundary
    dist_pred_to_target = dt_target[border_pred]
    # Distance from target boundary to prediction boundary
    dist_target_to_pred = dt_pred[border_target]

    combined = np.concatenate([dist_pred_to_target, dist_target_to_pred])
    hd95 = float(np.percentile(combined, 95))
    return float(np.clip(hd95, 0.0, max_penalty))


def evaluate_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    compute_contour_hd95: bool = False
) -> Dict[str, float]:
    """
    Evaluates multi-class segmentation for:
      - Liver (Class 1)
      - Tumor (Class 2)
      - Mean Average (Liver + Tumor)
      - Optional HD95 contour boundary error
    """
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    batch_size = preds.shape[0]
    liver_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}
    tumor_metrics = {"dice": [], "iou": [], "precision": [], "recall": []}
    hd95_liver = []
    hd95_tumor = []

    for b in range(batch_size):
        p_b = preds[b]
        t_b = targets_np[b]

        # Liver evaluation: Class 1 or 2 (any hepatic tissue)
        m_liver = compute_binary_metrics(p_b >= 1, t_b >= 1)
        # Tumor evaluation: Class 2 strictly
        m_tumor = compute_binary_metrics(p_b == 2, t_b == 2)

        for k in liver_metrics:
            liver_metrics[k].append(m_liver[k])
            tumor_metrics[k].append(m_tumor[k])

        if compute_contour_hd95:
            hd95_liver.append(compute_hd95(p_b >= 1, t_b >= 1))
            hd95_tumor.append(compute_hd95(p_b == 2, t_b == 2))

    results = {
        "liver_dice": float(np.mean(liver_metrics["dice"])),
        "liver_iou": float(np.mean(liver_metrics["iou"])),
        "liver_precision": float(np.mean(liver_metrics["precision"])),
        "liver_recall": float(np.mean(liver_metrics["recall"])),
        
        "tumor_dice": float(np.mean(tumor_metrics["dice"])),
        "tumor_iou": float(np.mean(tumor_metrics["iou"])),
        "tumor_precision": float(np.mean(tumor_metrics["precision"])),
        "tumor_recall": float(np.mean(tumor_metrics["recall"])),
    }

    # Macro mean of organ + lesion
    results["mean_dice"] = float((results["liver_dice"] + results["tumor_dice"]) / 2.0)
    results["mean_iou"] = float((results["liver_iou"] + results["tumor_iou"]) / 2.0)

    if compute_contour_hd95:
        results["liver_hd95"] = float(np.mean(hd95_liver))
        results["tumor_hd95"] = float(np.mean(hd95_tumor))
        results["mean_hd95"] = float((results["liver_hd95"] + results["tumor_hd95"]) / 2.0)

    return results

