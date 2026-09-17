"""
experiments/run_personalization_study.py
----------------------------------------
Flagship Comparative Study (Table 4 Reproduction):
Evaluates and contrasts the four core learning regimes on the exact same Non-IID split:
  1. Centralized Training (Pooled Upper Bound)
  2. Local-Only Training (Isolated Clients, No Collaboration)
  3. Standard FedAvg (Collaborative, Unpersonalized)
  4. FedAvg + CFL-Lite (Our Flagship: Clustered FL + Personalization Fine-Tuning)
Generates Table 4 equivalent metrics and publication-ready comparative bar plots.
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
from fl.cfl_simulation import run_cfl_lite_simulation
from model.unet_attention import AttentionUNetLite


def run_local_only_training(
    manifest_path: str = "data/processed/client_partitions.json",
    epochs: int = 3,
    batch_size: int = 4,
    base_filters: int = 16
) -> Dict[str, Any]:
    """
    Simulates isolated local training: each hospital trains strictly on its own private data.
    """
    print("\n--- Running Local-Only Baseline (No Federated Sharing) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    client_splits = manifest["non_iid_splits"]
    num_clients = len(client_splits)

    eval_results = []

    for cid_str, records in client_splits.items():
        cid = int(cid_str)
        n_train = int(0.80 * len(records))
        train_recs = records[:n_train]
        test_recs = records[n_train:]

        client = HospitalClient(
            client_id=cid,
            train_records=train_recs,
            test_records=test_recs,
            device=device,
            base_filters=base_filters,
            local_epochs=epochs,
            batch_size=batch_size
        )

        init_params = client.get_parameters({})
        trained_params, _, _ = client.fit(init_params, {"local_epochs": epochs})
        loss, n_test, metrics = client.evaluate(trained_params, {})
        eval_results.append((n_test, metrics))
        print(f"  Local Hospital {cid} -> Mean Dice: {metrics['mean_dice']:.4f} | Tumor Dice: {metrics['tumor_dice']:.4f}")

    aggregated = aggregate_eval_metrics(eval_results)
    return {
        "mean_dice": round(aggregated["mean_dice"], 4),
        "liver_dice": round(aggregated["liver_dice"], 4),
        "tumor_dice": round(aggregated["tumor_dice"], 4),
        "inter_client_std": round(aggregated["client_mean_dice_std"], 4),
        "details": aggregated
    }


def compile_table_4(results_dir: str = "results"):
    """
    Compiles all four conditions into Table 4 and renders comparison visualization.
    """
    print("\n" + "=" * 65)
    print("COMPILING TABLE 4: PERSONALIZATION BENCHMARK (NON-IID SKEW)")
    print("=" * 65)

    # 1. Centralized Upper Bound
    with open(os.path.join(results_dir, "centralized_benchmark.json"), "r") as f:
        centralized = json.load(f)

    # 2. Local-Only (run freshly on Non-IID split)
    local_only = run_local_only_training()

    # 3. Standard FedAvg (Non-IID)
    with open(os.path.join(results_dir, "fedavg_non_iid_benchmark.json"), "r") as f:
        fedavg_non_iid = json.load(f)

    # 4. CFL-Lite (Personalized)
    with open(os.path.join(results_dir, "cfl_lite_benchmark.json"), "r") as f:
        cfl_data = json.load(f)

    table_4 = {
        "Centralized Upper Bound": {
            "mean_dice": round(centralized["best_mean_dice"], 4),
            "liver_dice": round(centralized["final_metrics"]["liver_dice"], 4),
            "tumor_dice": round(centralized["final_metrics"]["tumor_dice"], 4),
            "inter_client_std": 0.0000,
            "description": "Pooled training upper bound"
        },
        "Local-Only (No FL)": {
            "mean_dice": local_only["mean_dice"],
            "liver_dice": local_only["liver_dice"],
            "tumor_dice": local_only["tumor_dice"],
            "inter_client_std": local_only["inter_client_std"],
            "description": "Isolated institutional training"
        },
        "FedAvg (Non-IID)": {
            "mean_dice": fedavg_non_iid["final_mean_dice"],
            "liver_dice": round(fedavg_non_iid["history"][-1]["liver_dice"], 4),
            "tumor_dice": fedavg_non_iid["final_tumor_dice"],
            "inter_client_std": fedavg_non_iid["final_inter_client_std"],
            "description": "Collaborative, unpersonalized baseline"
        },
        "FedAvg + CFL-Lite (Personalized)": {
            "mean_dice": cfl_data["final_mean_dice"],
            "liver_dice": round(cfl_data["history"][-1]["liver_dice"], 4),
            "tumor_dice": cfl_data["final_tumor_dice"],
            "inter_client_std": cfl_data["final_inter_client_std"],
            "description": "Our flagship clustered federated personalization"
        }
    }

    # Print Table 4 to console
    print(f"{'Method':<35} | {'Mean Dice':<10} | {'Liver Dice':<10} | {'Tumor Dice':<10} | {'Std (sigma)':<10}")
    print("-" * 85)
    for method, row in table_4.items():
        print(f"{method:<35} | {row['mean_dice']:<10.4f} | {row['liver_dice']:<10.4f} | {row['tumor_dice']:<10.4f} | {row['inter_client_std']:<10.4f}")
    print("-" * 85)

    # Save JSON
    output_json = os.path.join(results_dir, "table4_personalization_comparison.json")
    with open(output_json, "w") as f:
        json.dump(table_4, f, indent=2)
    print(f"[+] Table 4 saved to: {output_json}")

    # Render Visual Comparison Plot
    plot_path = os.path.join(results_dir, "table4_dice_and_variance.png")
    render_table_4_plot(table_4, plot_path)
    print(f"[+] Comparative bar plot saved to: {plot_path}")
    print("=" * 65)

    return table_4


def render_table_4_plot(table_4: Dict[str, Any], output_path: str):
    """
    Renders publication-ready comparative bar plot:
    Panel 1: Mean Dice across methods
    Panel 2: Inter-Client Standard Deviation (sigma) across methods
    """
    methods = list(table_4.keys())
    short_names = ["Centralized", "Local-Only", "FedAvg", "FedAvg + CFL-Lite\n(Personalized)"]
    dices = [table_4[m]["mean_dice"] for m in methods]
    stds = [table_4[m]["inter_client_std"] for m in methods]
    colors = ["#475569", "#dc2626", "#ea580c", "#16a34a"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))

    # Panel 1: Mean Dice
    bars1 = ax1.bar(short_names, dices, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.2)
    ax1.set_title("Clinical Mean Dice by Training Regime", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Dice Score", fontsize=11)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle=":", axis="y", alpha=0.6)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel 2: Inter-Client Standard Deviation (sigma)
    bars2 = ax2.bar(short_names[1:], stds[1:], color=colors[1:], width=0.55, edgecolor="#1e293b", linewidth=1.2)
    ax2.set_title(r"Inter-Client Disparity ($\sigma$ Std-Dev)", fontsize=12, fontweight="bold")
    ax2.set_ylabel(r"Standard Deviation ($\sigma$)", fontsize=11)
    ax2.grid(True, linestyle=":", axis="y", alpha=0.6)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    # 1. Run CFL-lite simulation
    run_cfl_lite_simulation(num_rounds=3, local_epochs=1, fine_tune_epochs=1, base_filters=16)
    # 2. Compile Table 4
    compile_table_4()
