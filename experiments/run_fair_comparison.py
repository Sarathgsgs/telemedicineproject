"""
experiments/run_fair_comparison.py
----------------------------------
Controlled Fair Model Comparison (Phase 2):
Benchmarks three learning configurations under strictly identical conditions:
  - Experiment A: Centralized Training (Pooled Upper Bound)
  - Experiment B: Standard FedAvg (Collaborative, Unpersonalized)
  - Experiment C: FedAvg + CFL-Lite (Clustered FL + 1 Local Fine-Tuning Epoch)

Controls enforced:
  1. Identical model architecture: AttentionUNetLite with GroupNorm (G=8)
  2. Identical starting weights and optimizer parameters (Adam, lr=1e-3, B=4)
  3. Identical patient-level Train/Val/Test partitioning (zero slice leakage)
  4. Identical loss formulation: CompositeLoss (Dice + Weighted CE + Focal)
  5. Evaluates Mean Dice, Liver Dice, Tumor Dice, and 95th Percentile Hausdorff Distance (HD95)
"""

import os
import sys
import json
import time
import torch
import numpy as np
from typing import Dict, Any, List

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.unet_attention import AttentionUNetLite
from model.losses import CompositeLoss
from model.metrics import evaluate_batch
from fl.client import HospitalClient, get_parameters, set_parameters
from fl.server import aggregate_eval_metrics
from fl.cfl_clustering import (
    flatten_parameter_updates,
    compute_cosine_similarity_matrix,
    cluster_clients_by_similarity
)


def run_fair_benchmark(
    manifest_path: str = "data/processed/real_lits/client_partitions_patient_level.json",
    num_rounds: int = 3,
    local_epochs: int = 1,
    base_filters: int = 16,
    batch_size: int = 4,
    seed: int = 42,
    results_dir: str = "results",
    out_filename: str = "fair_comparison_benchmark.json"
) -> Dict[str, Any]:
    # Fallback to synthetic partition if real_lits manifest does not exist
    if not os.path.exists(manifest_path):
        manifest_path = "data/processed/client_partitions.json"

    print("=" * 75)
    print("  PHASE 2: FAIR MODEL COMPARISON (CONTROLLED REPRODUCIBILITY)")
    print(f"  Dataset: {manifest_path}")
    print(f"  Rounds: {num_rounds} | Local Epochs: {local_epochs} | Batch Size: {batch_size} | Seed: {seed}")
    print("=" * 75)

    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load Partitions
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Support either patient-level format or legacy non-iid split format
    if "client_train_splits" in manifest:
        client_train = manifest["client_train_splits"]
        test_slices = manifest.get("test_slices", [])
        val_slices = manifest.get("val_slices", [])
    else:
        splits = manifest["non_iid_splits"]
        client_train = {}
        test_slices = []
        val_slices = []
        for cid_str, recs in splits.items():
            n_tr = int(0.80 * len(recs))
            client_train[cid_str] = recs[:n_tr]
            test_slices.extend(recs[n_tr:])

    print(f"[*] Training Clients: {len(client_train)} | Total Train Slices: {sum(len(v) for v in client_train.values())}")
    print(f"[*] Held-Out Test Slices: {len(test_slices)}")

    # 2. Initialize Seed Model (Identical Starting Weights for all 3 configurations)
    init_model = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=base_filters).to(device)
    base_params = get_parameters(init_model)

    comparison_results = {}
    eval_config = {"compute_hd95": True}

    # -------------------------------------------------------------
    # Experiment A: Centralized Upper Bound (Pooled Training Data)
    # -------------------------------------------------------------
    print("\n--- [Experiment A] Centralized Pooled Training (Upper Bound) ---")
    pooled_train = []
    for sl in client_train.values():
        pooled_train.extend(sl)

    central_client = HospitalClient(
        client_id=999,
        train_records=pooled_train,
        test_records=test_slices,
        device=device,
        base_filters=base_filters,
        local_epochs=num_rounds * local_epochs,
        batch_size=batch_size
    )
    c_params, _, _ = central_client.fit(base_params, {"local_epochs": num_rounds * local_epochs})
    loss_c, n_c, metrics_c = central_client.evaluate(c_params, eval_config)
    comparison_results["Centralized Baseline"] = {
        "mean_dice": round(metrics_c.get("mean_dice", 0.0), 4),
        "liver_dice": round(metrics_c.get("liver_dice", 0.0), 4),
        "tumor_dice": round(metrics_c.get("tumor_dice", 0.0), 4),
        "mean_hd95": round(metrics_c.get("mean_hd95", 0.0), 2),
        "tumor_hd95": round(metrics_c.get("tumor_hd95", 0.0), 2),
        "inter_client_std": 0.0000,
        "description": "Pooled training reference upper bound"
    }
    print(f"  Centralized -> Mean Dice: {metrics_c.get('mean_dice', 0.0):.4f} | Tumor Dice: {metrics_c.get('tumor_dice', 0.0):.4f} | HD95: {metrics_c.get('mean_hd95', 0.0):.2f} mm")

    # -------------------------------------------------------------
    # Experiment B: Standard FedAvg (Unpersonalized)
    # -------------------------------------------------------------
    print("\n--- [Experiment B] Standard FedAvg (Unpersonalized) ---")
    clients = []
    for cid_str, recs in client_train.items():
        c = HospitalClient(
            client_id=int(cid_str),
            train_records=recs,
            test_records=test_slices,
            device=device,
            base_filters=base_filters,
            local_epochs=local_epochs,
            batch_size=batch_size
        )
        clients.append(c)

    fedavg_params = [p.copy() for p in base_params]
    for rnd in range(1, num_rounds + 1):
        updates = []
        total_samples = 0
        for client in clients:
            w_new, n_s, _ = client.fit(fedavg_params, {"local_epochs": local_epochs})
            updates.append((w_new, n_s))
            total_samples += n_s

        # Coordinate-wise average
        fedavg_params = [
            sum((u[1] / max(total_samples, 1)) * u[0][l] for u in updates)
            for l in range(len(fedavg_params))
        ]

    # Evaluate FedAvg across clients
    eval_b = [c.evaluate(fedavg_params, eval_config) for c in clients]
    metrics_b = aggregate_eval_metrics([(e[1], e[2]) for e in eval_b])
    comparison_results["Standard FedAvg (Non-IID)"] = {
        "mean_dice": round(metrics_b.get("mean_dice", 0.0), 4),
        "liver_dice": round(metrics_b.get("liver_dice", 0.0), 4),
        "tumor_dice": round(metrics_b.get("tumor_dice", 0.0), 4),
        "mean_hd95": round(metrics_b.get("mean_hd95", 0.0), 2),
        "tumor_hd95": round(metrics_b.get("tumor_hd95", 0.0), 2),
        "inter_client_std": round(metrics_b.get("client_mean_dice_std", 0.0), 4),
        "description": "Standard unpersonalized collaborative averaging"
    }
    print(f"  FedAvg -> Mean Dice: {metrics_b.get('mean_dice', 0.0):.4f} | Tumor Dice: {metrics_b.get('tumor_dice', 0.0):.4f} | HD95: {metrics_b.get('mean_hd95', 0.0):.2f} mm | Inter-Client sigma: {metrics_b.get('client_mean_dice_std', 0.0):.4f}")

    # -------------------------------------------------------------
    # Experiment C: FedAvg + CFL-Lite (Personalized)
    # -------------------------------------------------------------
    print("\n--- [Experiment C] FedAvg + CFL-Lite (Personalized Flagship) ---")
    cfl_params = [p.copy() for p in base_params]
    personalized_params = {i: [p.copy() for p in base_params] for i in range(len(clients))}

    for rnd in range(1, num_rounds + 1):
        client_updates = []
        flat_deltas = []
        for cid, client in enumerate(clients):
            w_new, n_s, _ = client.fit(cfl_params, {"local_epochs": local_epochs})
            client_updates.append((w_new, n_s))
            flat_delta = flatten_parameter_updates(w_new, cfl_params)
            flat_deltas.append(flat_delta)

        # Pairwise update cosine similarity clustering
        S = compute_cosine_similarity_matrix(flat_deltas)
        clusters = cluster_clients_by_similarity(S, n_clusters=min(2, len(clients)))

        # Cluster aggregation + Local fine-tuning epoch
        for cid, client in enumerate(clients):
            assigned_c = next(c_id for c_id, members in clusters.items() if cid in members)
            cluster_members = clusters[assigned_c]
            c_weights = [
                np.mean([client_updates[m][0][l] for m in cluster_members], axis=0)
                for l in range(len(cfl_params))
            ]
            # 1 local fine-tuning epoch
            p_w, _, _ = client.fit(c_weights, {"local_epochs": 1})
            personalized_params[cid] = p_w

    eval_c = [clients[i].evaluate(personalized_params[i], eval_config) for i in range(len(clients))]
    metrics_cfl = aggregate_eval_metrics([(e[1], e[2]) for e in eval_c])
    comparison_results["FedAvg + CFL-Lite (Personalized)"] = {
        "mean_dice": round(metrics_cfl.get("mean_dice", 0.0), 4),
        "liver_dice": round(metrics_cfl.get("liver_dice", 0.0), 4),
        "tumor_dice": round(metrics_cfl.get("tumor_dice", 0.0), 4),
        "mean_hd95": round(metrics_cfl.get("mean_hd95", 0.0), 2),
        "tumor_hd95": round(metrics_cfl.get("tumor_hd95", 0.0), 2),
        "inter_client_std": round(metrics_cfl.get("client_mean_dice_std", 0.0), 4),
        "description": "Clustered federated learning + 1 local fine-tuning epoch"
    }
    print(f"  CFL-Lite -> Mean Dice: {metrics_cfl.get('mean_dice', 0.0):.4f} | Tumor Dice: {metrics_cfl.get('tumor_dice', 0.0):.4f} | HD95: {metrics_cfl.get('mean_hd95', 0.0):.2f} mm | Inter-Client sigma: {metrics_cfl.get('client_mean_dice_std', 0.0):.4f}")

    # 3. Print Final Comparison Table
    print("\n" + "=" * 88)
    print(f"{'Learning Paradigm':<32} | {'Mean Dice':<9} | {'Liver Dice':<10} | {'Tumor Dice':<10} | {'HD95 (mm)':<9} | {'sigma':<6}")
    print("-" * 88)
    for model_name, res in comparison_results.items():
        print(f"{model_name:<32} | {res['mean_dice']:<9.4f} | {res['liver_dice']:<10.4f} | {res['tumor_dice']:<10.4f} | {res['mean_hd95']:<9.2f} | {res['inter_client_std']:<6.4f}")
    print("=" * 88)

    out_file = os.path.join(results_dir, out_filename)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"[+] Benchmark results saved to: {out_file}")

    return comparison_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Controlled Fair Comparison Benchmark")
    default_manifest = "data/processed/real_lits/client_partitions_patient_level.json" if os.path.exists("data/processed/real_lits/client_partitions_patient_level.json") else "data/processed/client_partitions.json"
    parser.add_argument("--manifest", type=str, default=default_manifest,
                        help="Path to partition manifest JSON")
    parser.add_argument("--rounds", type=int, default=3, help="Federated communication rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--base-filters", type=int, default=16, help="Base filters for UNetLite")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="real_lits_fair_comparison.json", help="Output filename")
    args = parser.parse_args()

    run_fair_benchmark(
        manifest_path=args.manifest,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        base_filters=args.base_filters,
        batch_size=args.batch_size,
        seed=args.seed,
        out_filename=args.output
    )
