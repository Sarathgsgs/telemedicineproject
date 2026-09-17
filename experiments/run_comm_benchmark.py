"""
experiments/run_comm_benchmark.py
---------------------------------
Communication Efficiency Benchmark (Table 7 Reproduction):
Evaluates Top-k Gradient Sparsification with Error Feedback across multiple density levels:
  - Dense Baseline (100% density / 0% sparsity)
  - Moderate Sparsification (20% density / 80% sparsity)
  - High Sparsification (10% density / 90% sparsity)
  - Aggressive Sparsification (5% density / 95% sparsity)
Measures MB/round transmitted, bandwidth savings %, and clinical Dice retention.
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
from comm.topk_sparsify import TopKSparsifier
from model.unet_attention import AttentionUNetLite


def run_sparsified_federated_round(
    clients: List[HospitalClient],
    global_params: List[np.ndarray],
    sparsifier: TopKSparsifier
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Simulates one federated communication round with Top-k sparsification and error feedback.
    """
    num_clients = len(clients)
    client_updates = []
    total_samples = 0
    total_sparse_mb = 0.0
    total_dense_mb = 0.0

    # 1. Local client training & compression
    for cid, client in enumerate(clients):
        new_params, n_samples, _ = client.fit(global_params, {"local_epochs": 1})
        total_samples += n_samples

        # Calculate Delta w
        deltas = [new_p - glob_p for new_p, glob_p in zip(new_params, global_params)]

        # Apply Top-k compression with Error Feedback
        sparse_payload, meta = sparsifier.compress_updates(cid, deltas)
        total_sparse_mb += meta["sparse_mb"]
        total_dense_mb += meta["dense_mb"]

        # 2. Server decompresses sparse payload
        reconstructed_deltas = sparsifier.decompress_updates(sparse_payload)
        reconstructed_params = [glob_p + rec_d for glob_p, rec_d in zip(global_params, reconstructed_deltas)]

        client_updates.append((reconstructed_params, n_samples))

    # 3. Server-side Weighted Aggregation
    aggregated_params = []
    for l_idx in range(len(global_params)):
        layer_avg = sum((u[1] / total_samples) * u[0][l_idx] for u in client_updates)
        aggregated_params.append(layer_avg)

    # 4. Federated Evaluation across Local Hospital Splits
    eval_metrics = []
    for client in clients:
        loss, n_test, metrics = client.evaluate(aggregated_params, {})
        eval_metrics.append((n_test, metrics))

    summary = aggregate_eval_metrics(eval_metrics)
    summary["round_sparse_mb"] = round(total_sparse_mb, 3)
    summary["round_dense_mb"] = round(total_dense_mb, 3)
    summary["bandwidth_savings_pct"] = round((1.0 - total_sparse_mb / max(total_dense_mb, 0.001)) * 100, 1)

    return aggregated_params, summary


def run_communication_benchmark(
    manifest_path: str = "data/processed/client_partitions.json",
    density_levels: List[float] = [1.0, 0.20, 0.10, 0.05],
    num_rounds: int = 2,
    base_filters: int = 16,
    results_dir: str = "results"
):
    print("=" * 65)
    print("PHASE 7: COMMUNICATION EFFICIENCY BENCHMARK (TABLE 7)")
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
    base_params = get_parameters(init_model)

    benchmark_rows = []
    print(f"{'Regime':<24} | {'Density':<8} | {'MB/Round':<10} | {'Savings %':<10} | {'Mean Dice':<10} | {'Tumor Dice':<10}")
    print("-" * 85)

    for density in density_levels:
        sparsifier = TopKSparsifier(density_ratio=density)
        current_params = [p.copy() for p in base_params]
        final_summary = None

        for rnd in range(1, num_rounds + 1):
            current_params, final_summary = run_sparsified_federated_round(clients, current_params, sparsifier)

        regime_label = f"Dense Baseline" if density == 1.0 else f"Top-{int(density * 100)}% Sparsified"
        savings_str = f"{final_summary['bandwidth_savings_pct']:.1f}%" if density < 1.0 else "0.0% (Ref)"

        print(f"{regime_label:<24} | {density:<8.2f} | {final_summary['round_sparse_mb']:<10.2f} | {savings_str:<10} | {final_summary['mean_dice']:<10.4f} | {final_summary['tumor_dice']:<10.4f}")

        benchmark_rows.append({
            "regime": regime_label,
            "density_ratio": density,
            "sparsity_pct": round((1.0 - density) * 100, 1),
            "mb_per_round": final_summary["round_sparse_mb"],
            "total_mb_transferred": round(final_summary["round_sparse_mb"] * num_rounds, 2),
            "bandwidth_savings_pct": final_summary["bandwidth_savings_pct"],
            "mean_dice": round(final_summary["mean_dice"], 4),
            "liver_dice": round(final_summary["liver_dice"], 4),
            "tumor_dice": round(final_summary["tumor_dice"], 4)
        })

    print("-" * 85)

    # Save JSON (Table 7)
    table7_path = os.path.join(results_dir, "table7_comm_efficiency.json")
    with open(table7_path, "w") as f:
        json.dump(benchmark_rows, f, indent=2)
    print(f"[+] Table 7 saved to: {table7_path}")

    # Render Visual Comparison Plot
    plot_path = os.path.join(results_dir, "table7_comm_efficiency.png")
    render_table7_plot(benchmark_rows, plot_path)
    print(f"[+] Communication efficiency plot saved to: {plot_path}")
    print("=" * 65)

    return benchmark_rows


def render_table7_plot(benchmark_rows: List[Dict[str, Any]], output_path: str):
    """
    Renders publication-ready comparative bar/line plot:
    Panel 1: Bandwidth Transmitted per Round (MB) & Savings %
    Panel 2: Segmentation Accuracy Retention (Mean Dice)
    """
    regimes = [r["regime"].replace(" Sparsified", "") for r in benchmark_rows]
    mb_rounds = [r["mb_per_round"] for r in benchmark_rows]
    savings = [r["bandwidth_savings_pct"] for r in benchmark_rows]
    dices = [r["mean_dice"] for r in benchmark_rows]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel 1: Bandwidth per Round
    bars = ax1.bar(regimes, mb_rounds, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.1)
    ax1.set_title("Network Bandwidth per Global Round (MB)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Megabytes Transferred (MB / Round)", fontsize=11)
    ax1.grid(True, linestyle=":", axis="y", alpha=0.6)

    for bar, sav in zip(bars, savings):
        yval = bar.get_height()
        label = f"{yval:.1f} MB" if sav == 0.0 else f"{yval:.1f} MB\n(-{sav:.0f}%)"
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: Mean Dice Retention
    ax2.plot(regimes, dices, marker="o", linewidth=2.4, color="#2563eb", markersize=8)
    ax2.set_title("Clinical Mean Dice Retention under Sparsification", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Dice Score", fontsize=11)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle=":", alpha=0.6)

    for i, txt in enumerate(dices):
        ax2.annotate(f"{txt:.4f}", (regimes[i], dices[i] + 0.03), ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    run_communication_benchmark()
