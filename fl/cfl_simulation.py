"""
fl/cfl_simulation.py
--------------------
Flagship Personalization Orchestrator:
Executes Clustered Federated Learning (CFL-Lite) with Local Adaptation on Non-IID clinical data.
Clusters hospitals by gradient/weight cosine similarity, aggregates cluster-wise,
and performs local fine-tuning to recover non-IID accuracy and reduce inter-client variance.
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, List, Any, Tuple

from fl.client import HospitalClient, get_parameters, set_parameters
from fl.server import aggregate_eval_metrics
from fl.cfl_clustering import (
    flatten_parameter_updates,
    compute_cosine_similarity_matrix,
    cluster_clients_by_similarity,
    aggregate_within_clusters
)
from model.unet_attention import AttentionUNetLite


def run_cfl_lite_simulation(
    manifest_path: str = "data/processed/client_partitions.json",
    num_rounds: int = 3,
    local_epochs: int = 1,
    fine_tune_epochs: int = 1,
    batch_size: int = 4,
    base_filters: int = 16,
    n_clusters: int = 2,
    results_dir: str = "results",
    checkpoints_dir: str = "checkpoints"
) -> Dict[str, Any]:
    print("=" * 65)
    print("PHASE 5: CLUSTERED FEDERATED LEARNING (CFL-LITE) WITH PERSONALIZATION")
    print("=" * 65)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # 1. Load Non-IID Partitions
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    client_splits = manifest["non_iid_splits"]
    num_clients = len(client_splits)

    clients = []
    print(f"[*] Initializing {num_clients} simulated hospitals under Non-IID Skew...")
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

    # 2. Global Initialization
    init_model = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=base_filters).to(device)
    global_params = get_parameters(init_model)

    # Initial cluster mapping: All in single cluster for warm-up round 1
    cluster_mapping = {0: list(range(num_clients))}
    cluster_params = {0: global_params}

    history = []
    start_time = time.time()
    similarity_matrices = []

    # 3. Federated Communication Rounds
    for rnd in range(1, num_rounds + 1):
        round_start = time.time()
        print(f"\n>>> CFL-Lite Global Round [{rnd:02d}/{num_rounds:02d}] <<<")

        # --- A. Local Training with Cluster Models ---
        client_updates = []
        client_deltas = []
        total_samples = 0

        for cid, client in enumerate(clients):
            # Find which cluster this client belongs to
            assigned_cluster = next(c_id for c_id, members in cluster_mapping.items() if cid in members)
            model_to_send = cluster_params[assigned_cluster]

            # Fit on client data
            fit_res = client.fit(model_to_send, {"local_epochs": local_epochs})
            new_params, n_samples, _ = fit_res
            client_updates.append((new_params, n_samples))
            total_samples += n_samples

            # Compute update vector Delta w
            delta_vec = flatten_parameter_updates(new_params, model_to_send)
            client_deltas.append(delta_vec)

        # --- B. Cosine Similarity Matrix & Cluster Re-evaluation ---
        sim_matrix = compute_cosine_similarity_matrix(client_deltas)
        similarity_matrices.append(sim_matrix.tolist())

        if rnd >= 2:
            cluster_mapping = cluster_clients_by_similarity(sim_matrix, n_clusters=n_clusters)
            print(f"  [CFL Dynamic Clustering] Round {rnd} Assignments: {cluster_mapping}")
        else:
            print(f"  [Warm-up Round 1] Single global cluster maintained.")

        # --- C. Cluster-Wise Aggregation ---
        cluster_params = aggregate_within_clusters(cluster_mapping, client_updates)

        # --- D. Flagship Step: Local Fine-Tuning (Personalization) ---
        personalized_params = {}
        for cid, client in enumerate(clients):
            assigned_cluster = next(c_id for c_id, members in cluster_mapping.items() if cid in members)
            c_agg_params = cluster_params[assigned_cluster]

            # 1 fine-tuning epoch on local client data to personalize features
            ft_res = client.fit(c_agg_params, {"local_epochs": fine_tune_epochs})
            p_params, _, _ = ft_res
            personalized_params[cid] = p_params

        # --- E. Evaluate Personalized Models on Local Held-Out Splits ---
        eval_metrics = []
        for cid, client in enumerate(clients):
            loss, n_test, c_metrics = client.evaluate(personalized_params[cid], {})
            eval_metrics.append((n_test, c_metrics))

        round_summary = aggregate_eval_metrics(eval_metrics)
        round_elapsed = time.time() - round_start

        print(f"  [Personalized Round {rnd:02d} Summary ({round_elapsed:.1f}s)] "
              f"Mean Dice: {round_summary['mean_dice']:.4f} | "
              f"Liver Dice: {round_summary['liver_dice']:.4f} | "
              f"Tumor Dice: {round_summary['tumor_dice']:.4f} | "
              f"Inter-Client Std (sigma): {round_summary['client_mean_dice_std']:.4f}")

        history.append({
            "round": rnd,
            "round_time_sec": round(round_elapsed, 2),
            "cluster_mapping": {str(k): v for k, v in cluster_mapping.items()},
            **round_summary
        })

    total_time = time.time() - start_time
    print(f"\n[+] CFL-Lite simulation complete in {total_time:.1f}s.")

    # 4. Save Final Results
    final_benchmark = {
        "method": "CFL-Lite (Clustered FL + Personalization Fine-Tuning)",
        "num_clients": num_clients,
        "n_clusters": n_clusters,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "total_time_seconds": round(total_time, 1),
        "best_mean_dice": round(max(h["mean_dice"] for h in history), 4),
        "final_mean_dice": round(history[-1]["mean_dice"], 4),
        "final_tumor_dice": round(history[-1]["tumor_dice"], 4),
        "final_inter_client_std": round(history[-1]["client_mean_dice_std"], 4),
        "final_clusters": {str(k): v for k, v in cluster_mapping.items()},
        "history": history,
        "similarity_matrices": similarity_matrices
    }

    output_json = os.path.join(results_dir, "cfl_lite_benchmark.json")
    with open(output_json, "w") as f:
        json.dump(final_benchmark, f, indent=2)
    print(f"[+] Saved CFL benchmark to: {output_json}")

    # Save best personalized checkpoints
    for cid in range(num_clients):
        set_parameters(init_model, personalized_params[cid])
        ckpt_path = os.path.join(checkpoints_dir, f"cfl_client_{cid}_personalized.pt")
        torch.save(init_model.state_dict(), ckpt_path)
    print(f"[+] Saved per-client personalized checkpoints to: {checkpoints_dir}")
    print("=" * 65)

    return final_benchmark
