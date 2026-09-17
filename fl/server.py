"""
fl/server.py
------------
Federated Learning Aggregation Strategy for FedLiverNet.
Implements dataset-size weighted FedAvg aggregation, per-round metric aggregation,
and tracks cross-client variance (standard deviation sigma) across institutions.
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)


def aggregate_eval_metrics(metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
    """
    Computes dataset-size weighted average and inter-client standard deviation (sigma)
    across all hospital nodes.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    keys = [
        "liver_dice", "tumor_dice", "mean_dice",
        "liver_iou", "tumor_iou",
        "liver_precision", "tumor_precision",
        "liver_recall", "tumor_recall"
    ]

    aggregated = {}
    client_mean_dices = []
    client_tumor_dices = []

    for k in keys:
        weighted_val = sum(num_examples * m.get(k, 0.0) for num_examples, m in metrics)
        aggregated[k] = float(weighted_val / total_examples)

    # Calculate inter-client variance / standard deviation (sigma)
    for _, m in metrics:
        if "mean_dice" in m:
            client_mean_dices.append(float(m["mean_dice"]))
        if "tumor_dice" in m:
            client_tumor_dices.append(float(m["tumor_dice"]))

    aggregated["client_mean_dice_std"] = float(np.std(client_mean_dices)) if client_mean_dices else 0.0
    aggregated["client_tumor_dice_std"] = float(np.std(client_tumor_dices)) if client_tumor_dices else 0.0
    aggregated["num_participating_clients"] = len(metrics)

    return aggregated


class FedAvgWithVariance(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that logs inter-client variance (sigma)
    and saves global checkpoint states.
    """
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        run_name: str = "fedavg",
        **kwargs
    ):
        super().__init__(
            evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
            **kwargs
        )
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.best_dice = 0.0
        self.history = []
        os.makedirs(checkpoint_dir, exist_ok=True)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregates evaluation results from all clients and logs inter-client variance.
        """
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        if metrics_aggregated:
            print(f"[Round {server_round:02d} Aggregated] "
                  f"Mean Dice: {metrics_aggregated.get('mean_dice', 0.0):.4f} | "
                  f"Tumor Dice: {metrics_aggregated.get('tumor_dice', 0.0):.4f} | "
                  f"Inter-Client Std (sigma): {metrics_aggregated.get('client_mean_dice_std', 0.0):.4f}")

            self.history.append({
                "round": server_round,
                "loss": loss_aggregated,
                **metrics_aggregated
            })

            current_dice = metrics_aggregated.get("mean_dice", 0.0)
            if current_dice > self.best_dice:
                self.best_dice = current_dice

        return loss_aggregated, metrics_aggregated
