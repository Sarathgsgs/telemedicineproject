"""
experiments/run_privacy_sweep.py
--------------------------------
Differential Privacy Budget (epsilon) vs Utility (Dice Score) Sweep:
Replicates FedLiverNet Table 5 / Fig. 8 privacy-utility tradeoff curve.
Sweeps epsilon in {0.1, 0.5, 1.0, 5.0, infinity} and tests Zero-Sum Additive Masking.
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from fl.client import HospitalClient, get_parameters, set_parameters
from fl.server import aggregate_eval_metrics
from privacy.dp_mechanism import apply_differential_privacy
from privacy.additive_masking import AdditiveMaskingManager
from model.unet_attention import AttentionUNetLite


def run_privacy_evaluation_round(
    clients: List[HospitalClient],
    global_params: List[np.ndarray],
    epsilon: float,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    use_masking: bool = True
) -> Dict[str, Any]:
    """
    Executes a federated round under client-side DP noise injection and additive masking.
    """
    num_clients = len(clients)
    layer_shapes = [p.shape for p in global_params]

    # Initialize Additive Masking Manager
    mask_mgr = AdditiveMaskingManager(num_clients, layer_shapes, seed=123) if use_masking else None

    client_updates = []
    total_samples = 0
    noise_sigma = 0.0

    # 1. Local Training + Client-side DP
    for cid, client in enumerate(clients):
        new_params, n_samples, _ = client.fit(global_params, {"local_epochs": 1})
        total_samples += n_samples

        # Apply Differential Privacy (L2 clipping + calibrated Gaussian noise)
        noisy_params, dp_meta = apply_differential_privacy(
            parameter_updates=new_params,
            epsilon=epsilon,
            delta=delta,
            clip_norm=clip_norm,
            seed=42 + cid
        )
        noise_sigma = dp_meta["sigma"]

        # Apply Zero-Sum Additive Masking
        if use_masking:
            transmitted_params = mask_mgr.mask_client_parameters(cid, noisy_params)
        else:
            transmitted_params = noisy_params

        client_updates.append((transmitted_params, n_samples))

    # 2. Server-side Aggregation
    if use_masking:
        # Server aggregates masked parameters; all pairwise masks cancel out exactly to 0
        raw_masked = [u[0] for u in client_updates]
        unmasked_sum = mask_mgr.aggregate_masked_parameters(raw_masked)
        aggregated_params = [layer / num_clients for layer in unmasked_sum]
    else:
        aggregated_params = []
        for l_idx in range(len(global_params)):
            layer_avg = sum((u[1] / total_samples) * u[0][l_idx] for u in client_updates)
            aggregated_params.append(layer_avg)

    # 3. Federated Evaluation across Local Hospital Splits
    eval_metrics = []
    for client in clients:
        loss, n_test, metrics = client.evaluate(aggregated_params, {})
        eval_metrics.append((n_test, metrics))

    summary = aggregate_eval_metrics(eval_metrics)
    summary["epsilon"] = epsilon if epsilon != float("inf") else "infinity"
    summary["noise_sigma"] = round(noise_sigma, 6)
    summary["additive_masking_active"] = use_masking

    return summary


def run_full_privacy_sweep(
    manifest_path: str = "data/processed/client_partitions.json",
    epsilons: List[float] = [0.1, 0.5, 1.0, 5.0, float("inf")],
    base_filters: int = 16,
    results_dir: str = "results"
):
    print("=" * 65)
    print("PHASE 6: PRIVACY MECHANISMS & EPSILON BUDGET SWEEP (TABLE 5)")
    print("=" * 65)

    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    client_splits = manifest["non_iid_splits"]
    clients = []
    for cid_str, records in client_splits.items():
        cid = int(cid_str)
        n_train = int(0.80 * len(records))
        client = HospitalClient(
            client_id=cid,
            train_records=records[:n_train],
            test_records=records[n_train:],
            device=device,
            base_filters=base_filters,
            local_epochs=1,
            batch_size=4
        )
        clients.append(client)

    init_model = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=base_filters).to(device)
    global_params = get_parameters(init_model)

    sweep_results = []
    print(f"{'Epsilon (eps)':<15} | {'Noise Sigma':<14} | {'Mean Dice':<12} | {'Liver Dice':<12} | {'Tumor Dice':<12}")
    print("-" * 75)

    for eps in epsilons:
        res = run_privacy_evaluation_round(
            clients=clients,
            global_params=global_params,
            epsilon=eps,
            delta=1e-5,
            clip_norm=1.0,
            use_masking=True
        )
        eps_str = str(eps) if eps != float("inf") else "inf (No DP)"
        print(f"{eps_str:<15} | {res['noise_sigma']:<14.6f} | {res['mean_dice']:<12.4f} | {res['liver_dice']:<12.4f} | {res['tumor_dice']:<12.4f}")
        sweep_results.append({
            "epsilon_label": eps_str,
            "epsilon_value": eps if eps != float("inf") else 999.0,
            **res
        })

    print("-" * 75)

    # Save JSON Benchmark (Table 5)
    table5_path = os.path.join(results_dir, "table5_privacy_sweep.json")
    with open(table5_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[+] Table 5 results saved to: {table5_path}")

    # Render Privacy-Utility Pareto Curve
    plot_path = os.path.join(results_dir, "table5_privacy_utility_tradeoff.png")
    render_privacy_curve(sweep_results, plot_path)
    print(f"[+] Privacy-Utility Tradeoff plot saved to: {plot_path}")
    print("=" * 65)

    return sweep_results


def render_privacy_curve(sweep_results: List[Dict[str, Any]], output_path: str):
    """
    Renders the classic Privacy-Utility Tradeoff curve matching Fig. 8 in FedLiverNet.
    """
    labels = [r["epsilon_label"] for r in sweep_results]
    mean_dices = [r["mean_dice"] for r in sweep_results]
    liver_dices = [r["liver_dice"] for r in sweep_results]
    tumor_dices = [r["tumor_dice"] for r in sweep_results]
    sigmas = [r["noise_sigma"] for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel 1: Privacy vs Dice Score
    x = np.arange(len(labels))
    ax1.plot(x, mean_dices, marker='o', linewidth=2.4, color='#2563eb', label='Mean Dice Score')
    ax1.plot(x, liver_dices, marker='^', linewidth=2.0, color='#16a34a', linestyle='--', label='Liver Dice')
    ax1.plot(x, tumor_dices, marker='s', linewidth=2.0, color='#dc2626', linestyle='-.', label='Tumor Dice')
    ax1.set_title("Privacy-Utility Tradeoff Curve (FedLiverNet Fig. 8)", fontsize=12, fontweight='bold')
    ax1.set_xlabel(r"Privacy Budget ($\epsilon$ - Differential Privacy)", fontsize=11)
    ax1.set_ylabel("Clinical Segmentation Dice", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True)

    # Panel 2: Calibrated Gaussian Noise Sigma
    bars = ax2.bar(x, sigmas, color='#8b5cf6', width=0.5, edgecolor='#4c1d95', linewidth=1.1)
    ax2.set_title(r"Injected Calibrated Gaussian Noise ($\sigma$)", fontsize=12, fontweight='bold')
    ax2.set_xlabel(r"Privacy Budget ($\epsilon$)", fontsize=11)
    ax2.set_ylabel(r"Noise Standard Deviation ($\sigma$)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(True, linestyle=':', axis='y', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    run_full_privacy_sweep()
