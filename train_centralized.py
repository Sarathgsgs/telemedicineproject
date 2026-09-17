"""
train_centralized.py
--------------------
Centralized baseline model training on pooled (non-federated) dataset.
Acts as the upper-bound benchmark for Table 4 (Centralized vs FedAvg vs CFL-lite).
Trains Attention U-Net-lite with GroupNorm using Composite Loss (Dice + CE + Focal).
"""

import os
import json
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.unet_attention import AttentionUNetLite
from model.losses import CompositeLoss
from model.metrics import evaluate_batch
from data.dataset import get_dataloader


def run_centralized_training(
    manifest_path: str = "data/processed/client_partitions.json",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    output_dir: str = "checkpoints",
    results_dir: str = "results"
):
    print("=" * 65)
    print("PHASE 3: CENTRALIZED BASELINE TRAINING (POOLED DATASET)")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # 1. Load Data Splits
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run Phase 2 first.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Pool all client records
    all_records = []
    for cid_str, split_records in manifest["non_iid_splits"].items():
        all_records.extend(split_records)

    # 80% Train, 20% Held-out Test
    np.random.seed(42)
    indices = np.random.permutation(len(all_records))
    train_size = int(0.80 * len(all_records))

    train_records = [all_records[i] for i in indices[:train_size]]
    test_records = [all_records[i] for i in indices[train_size:]]

    print(f"[*] Pooled Dataset: {len(all_records)} slices | Train: {len(train_records)} | Test: {len(test_records)}")

    train_loader = get_dataloader(train_records, batch_size=batch_size, shuffle=True, augment=True)
    test_loader = get_dataloader(test_records, batch_size=batch_size, shuffle=False, augment=False)

    # 2. Initialize Model & Composite Loss
    model = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=32).to(device)
    total_params = model.count_parameters()
    print(f"[+] Model: AttentionUNetLite | Parameters: {total_params:,} (~{total_params / 1e6:.2f}M)")

    criterion = CompositeLoss(alpha=1.0, beta=0.5, gamma=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_mean_dice = 0.0
    history = []

    # 3. Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss, _ = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            metrics = evaluate_batch(logits, masks)
            train_dice += metrics["mean_dice"] * images.size(0)

        scheduler.step()
        train_loss /= len(train_records)
        train_dice /= len(train_records)

        # 4. Evaluation Loop
        model.eval()
        test_loss = 0.0
        test_metrics_sum = {
            "liver_dice": 0.0, "tumor_dice": 0.0, "mean_dice": 0.0,
            "liver_iou": 0.0, "tumor_iou": 0.0, "mean_iou": 0.0,
            "liver_precision": 0.0, "tumor_precision": 0.0,
            "liver_recall": 0.0, "tumor_recall": 0.0
        }

        with torch.no_grad():
            for images, masks, _ in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss, _ = criterion(logits, masks)
                test_loss += loss.item() * images.size(0)

                b_metrics = evaluate_batch(logits, masks)
                for k in test_metrics_sum:
                    test_metrics_sum[k] += b_metrics[k] * images.size(0)

        test_loss /= len(test_records)
        epoch_eval = {k: v / len(test_records) for k, v in test_metrics_sum.items()}

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Liver Dice: {epoch_eval['liver_dice']:.4f} | "
              f"Tumor Dice: {epoch_eval['tumor_dice']:.4f} | "
              f"Mean Dice: {epoch_eval['mean_dice']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            **epoch_eval
        })

        if epoch_eval["mean_dice"] > best_mean_dice:
            best_mean_dice = epoch_eval["mean_dice"]
            best_checkpoint_path = os.path.join(output_dir, "centralized_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_eval,
                "parameters": total_params
            }, best_checkpoint_path)

    elapsed = time.time() - start_time
    print(f"\n[+] Centralized training complete in {elapsed:.1f}s. Best Mean Dice: {best_mean_dice:.4f}")

    # 5. Save Benchmark JSON
    final_benchmark = {
        "architecture": "2D Attention U-Net-lite (GroupNorm)",
        "parameters": total_params,
        "epochs": epochs,
        "training_time_seconds": round(elapsed, 1),
        "best_mean_dice": round(best_mean_dice, 4),
        "final_metrics": history[-1],
        "training_history": history
    }

    benchmark_file = os.path.join(results_dir, "centralized_benchmark.json")
    with open(benchmark_file, "w") as f:
        json.dump(final_benchmark, f, indent=2)
    print(f"[+] Benchmark results saved to: {benchmark_file}")

    # 6. Qualitative Prediction Plot
    print("[*] Generating qualitative test prediction overlays...")
    qual_plot_path = os.path.join(results_dir, "centralized_test_predictions.png")
    generate_prediction_plot(model, test_records[:3], device, qual_plot_path)
    print(f"[+] Qualitative visualization saved to: {qual_plot_path}")
    print("=" * 65)

    return final_benchmark, best_checkpoint_path, qual_plot_path


def generate_prediction_plot(model, test_records, device, output_path: str):
    """
    Renders 4-column qualitative evaluation plot:
    [Raw CT Slice] | [Ground Truth Overlay] | [Attention Gate Map] | [Model Prediction Overlay]
    """
    model.eval()
    num_samples = len(test_records)
    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.5 * num_samples))

    with torch.no_grad():
        for i, rec in enumerate(test_records):
            with np.load(rec["filepath"]) as d:
                raw_img = d["image"]
                gt_mask = d["mask"]

            img_tensor = torch.from_numpy(raw_img).unsqueeze(0).unsqueeze(0).float().to(device)
            logits, att_maps = model(img_tensor, return_attention_maps=True)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            # Finest attention gate (Level 1)
            att_map = att_maps[0].squeeze().cpu().numpy()

            # Ground Truth Overlay
            gt_overlay = np.stack([raw_img, raw_img, raw_img], axis=-1)
            gt_overlay[gt_mask == 1, 1] = np.clip(gt_overlay[gt_mask == 1, 1] * 0.4 + 0.6, 0, 1)
            gt_overlay[gt_mask == 2] = [1.0, 0.15, 0.15]

            # Prediction Overlay
            pred_overlay = np.stack([raw_img, raw_img, raw_img], axis=-1)
            pred_overlay[pred_mask == 1, 1] = np.clip(pred_overlay[pred_mask == 1, 1] * 0.4 + 0.6, 0, 1)
            pred_overlay[pred_mask == 2] = [1.0, 0.15, 0.15]

            cat = rec.get("category", "test")

            # Column 1: Raw
            axes[i, 0].imshow(raw_img, cmap="gray")
            axes[i, 0].set_title(f"Input CT Slice\n[{cat}]", fontsize=10)
            axes[i, 0].axis("off")

            # Column 2: Ground Truth
            axes[i, 1].imshow(gt_overlay)
            axes[i, 1].set_title("Ground Truth\n(Green: Liver | Red: Tumor)", fontsize=10)
            axes[i, 1].axis("off")

            # Column 3: Attention Gate Map
            axes[i, 2].imshow(att_map, cmap="inferno")
            axes[i, 2].set_title("Attention Gate Activation\n(Focus on Salient Tissue)", fontsize=10)
            axes[i, 2].axis("off")

            # Column 4: Model Prediction
            axes[i, 3].imshow(pred_overlay)
            axes[i, 3].set_title("Predicted Segmentation\n(Attention U-Net-Lite)", fontsize=10)
            axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    run_centralized_training(epochs=5, batch_size=8)
