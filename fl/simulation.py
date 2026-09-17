"""
fl/simulation.py
----------------
Federated Learning Simulation Orchestrator for FedLiverNet.
Executes decentralized FedAvg training under both IID and Non-IID clinical splits.
Generates comparative convergence curves and inter-client standard deviation (sigma)
reproducing the baseline drop that motivates the Phase 5 CFL-lite personalization flagship.
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from fl.client import HospitalClient, get_parameters, set_parameters
from fl.server import aggregate_eval_metrics
from model.unet_attention import AttentionUNetLite


def run_federated_simulation(
    split_type: str = "non_iid",
    manifest_path: str = "data/processed/client_partitions.json",
    num_rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 4,
    base_filters: int = 32,
    results_dir: str = "results",
    checkpoints_dir: str = "checkpoints"
) -> Dict[str, Any]:
    """
    Executes federated rounds across simulated hospital clients.
    
    Args:
        split_type: 'iid' or 'non_iid'
        num_rounds: Number of global communication rounds.
        local_epochs: Number of local SGD epochs per client per round.
    """
    print("=" * 65)
    print(f"PHASE 4: FEDERATED LEARNING SIMULATION (FedAvg - {split_type.upper()})")
    print("=" * 65)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # 1. Load Partitions
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    split_key = "iid_splits" if split_type == "iid" else "non_iid_splits"
    client_splits = manifest[split_key]
    num_clients = len(client_splits)

    # Instantiate Hospital Clients (80% Train, 20% Local Test per hospital)
    clients = []
    print(f"[*] Initializing {num_clients} simulated hospital clients ({split_type.upper()})...")
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
            local_epochs=local_epochs,
            batch_size=batch_size
        )
        clients.append(client)
        print(f"  - Client {cid}: {len(train_recs)} train, {len(test_recs)} test slices.")

    # 2. Initialize Global Model Parameters
    global_model = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=base_filters).to(device)
    global_params = get_parameters(global_model)

    history = []
    start_time = time.time()

    # 3. Federated Communication Rounds
    for rnd in range(1, num_rounds + 1):
        round_start = time.time()
        print(f"\n>>> Global Round [{rnd:02d}/{num_rounds:02d}] <<<")

        # --- A. Local Client Training ---
        client_updates = []
        total_train_samples = 0

        for client in clients:
            # Send current global model to client
            fit_res = client.fit(global_params, {"local_epochs": local_epochs})
            new_params, num_samples, metrics = fit_res
            client_updates.append((new_params, num_samples))
            total_train_samples += num_samples

        # --- B. Server Weighted FedAvg Aggregation ---
        # w_{t+1} = sum_k (n_k / n) * w_{t+1}^k
        new_global_params = []
        for param_idx in range(len(global_params)):
            weighted_param = sum(
                (num_samples / total_train_samples) * client_params[param_idx]
                for client_params, num_samples in client_updates
            )
            new_global_params.append(weighted_param)
        global_params = new_global_params

        # --- C. Federated Evaluation across Local Held-Out Hospital Splits ---
        eval_metrics = []
        for client in clients:
            loss, num_test, c_metrics = client.evaluate(global_params, {})
            eval_metrics.append((num_test, c_metrics))

        # Server Aggregation & Variance Tracking
        round_summary = aggregate_eval_metrics(eval_metrics)
        round_elapsed = time.time() - round_start

        print(f"  [Round {rnd:02d} Summary ({round_elapsed:.1f}s)] "
              f"Mean Dice: {round_summary['mean_dice']:.4f} | "
              f"Liver Dice: {round_summary['liver_dice']:.4f} | "
              f"Tumor Dice: {round_summary['tumor_dice']:.4f} | "
              f"Inter-Client Std (sigma): {round_summary['client_mean_dice_std']:.4f}")

        history.append({
            "round": rnd,
            "round_time_sec": round(round_elapsed, 2),
            **round_summary
        })

    total_time = time.time() - start_time
    print(f"\n[+] Federated simulation ({split_type}) completed in {total_time:.1f}s.")

    # 4. Save Benchmark JSON
    final_benchmark = {
        "split_type": split_type,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "total_time_seconds": round(total_time, 1),
        "best_mean_dice": round(max(h["mean_dice"] for h in history), 4),
        "final_mean_dice": round(history[-1]["mean_dice"], 4),
        "final_tumor_dice": round(history[-1]["tumor_dice"], 4),
        "final_inter_client_std": round(history[-1]["client_mean_dice_std"], 4),
        "history": history
    }

    output_json = os.path.join(results_dir, f"fedavg_{split_type}_benchmark.json")
    with open(output_json, "w") as f:
        json.dump(final_benchmark, f, indent=2)
    print(f"[+] Saved benchmark to: {output_json}")

    # Save checkpoint
    checkpoint_file = os.path.join(checkpoints_dir, f"fedavg_{split_type}_final.pt")
    set_parameters(global_model, global_params)
    torch.save(global_model.state_dict(), checkpoint_file)
    print(f"[+] Saved global checkpoint to: {checkpoint_file}")
    print("=" * 65)

    return final_benchmark


def plot_iid_vs_non_iid_comparison(
    iid_results_path: str = "results/fedavg_iid_benchmark.json",
    non_iid_results_path: str = "results/fedavg_non_iid_benchmark.json",
    output_plot_path: str = "results/fedavg_iid_vs_non_iid.png"
):
    """
    Renders comparative convergence plot:
    Panel 1: Mean Dice Convergence across rounds (IID vs Non-IID)
    Panel 2: Inter-Client Standard Deviation (sigma) across rounds (IID vs Non-IID)
    Directly mirrors the base paper's Fig. 7 narrative.
    """
    with open(iid_results_path, "r") as f:
        iid_data = json.load(f)
    with open(non_iid_results_path, "r") as f:
        non_iid_data = json.load(f)

    rounds = [h["round"] for h in iid_data["history"]]
    iid_dice = [h["mean_dice"] for h in iid_data["history"]]
    non_iid_dice = [h["mean_dice"] for h in non_iid_data["history"]]

    iid_std = [h["client_mean_dice_std"] for h in iid_data["history"]]
    non_iid_std = [h["client_mean_dice_std"] for h in non_iid_data["history"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel 1: Convergence
    ax1.plot(rounds, iid_dice, marker='o', linewidth=2.2, color='#2563eb', label='FedAvg (IID Split)')
    ax1.plot(rounds, non_iid_dice, marker='s', linewidth=2.2, color='#dc2626', linestyle='--', label='FedAvg (Non-IID Skew)')
    ax1.set_title("Federated Convergence: IID vs Non-IID Skew", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Global Communication Round", fontsize=11)
    ax1.set_ylabel("Mean Dice Score", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True)

    # Panel 2: Variance
    x = np.arange(len(rounds))
    width = 0.35
    ax2.bar(x - width/2, iid_std, width, label='IID Inter-Client Std', color='#60a5fa')
    ax2.bar(x + width/2, non_iid_std, width, label='Non-IID Inter-Client Std', color='#f87171')
    ax2.set_title(r"Inter-Client Variance ($\sigma$): Non-IID Disparity", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Global Communication Round", fontsize=11)
    ax2.set_ylabel(r"Standard Deviation ($\sigma$)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(rounds)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=180)
    plt.close()
    print(f"[+] Comparative plot saved to: {output_plot_path}")


if __name__ == "__main__":
    # Run IID first
    run_federated_simulation(split_type="iid", num_rounds=3, local_epochs=1, base_filters=16)
    # Run Non-IID next
    run_federated_simulation(split_type="non_iid", num_rounds=3, local_epochs=1, base_filters=16)
    # Compare
    plot_iid_vs_non_iid_comparison()
