"""
fl/client.py
------------
Flower NumPyClient for federated liver and lesion segmentation.
Implements decentralized local training, parameter extraction,
and local clinical metric evaluation (Dice, IoU, Precision, Recall).
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import OrderedDict
import flwr as fl

from model.unet_attention import AttentionUNetLite
from model.losses import CompositeLoss
from model.metrics import evaluate_batch
from data.dataset import get_dataloader


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extracts PyTorch model weights as an independent cloned list of NumPy arrays."""
    return [val.detach().cpu().clone().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]):
    """Loads a list of NumPy weight arrays into PyTorch model state_dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class HospitalClient(fl.client.NumPyClient):
    """
    Simulated Hospital Institution Client for Federated Learning.
    Trains on local private CT slices without exposing patient imaging data.
    """
    def __init__(
        self,
        client_id: int,
        train_records: List[Dict[str, Any]],
        test_records: List[Dict[str, Any]],
        device: torch.device = None,
        base_filters: int = 32,
        local_epochs: int = 1,
        batch_size: int = 4,
        lr: float = 1e-3
    ):
        self.client_id = client_id
        self.train_records = train_records
        self.test_records = test_records
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.model = AttentionUNetLite(
            in_channels=1,
            num_classes=3,
            base_filters=base_filters
        ).to(self.device)

        self.criterion = CompositeLoss(alpha=1.0, beta=0.5, gamma=0.5).to(self.device)
        self.train_loader = get_dataloader(self.train_records, batch_size=self.batch_size, shuffle=True, augment=True)
        self.test_loader = get_dataloader(self.test_records, batch_size=self.batch_size, shuffle=False, augment=False)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Executes local training on this hospital's private dataset.
        """
        set_parameters(self.model, parameters)

        epochs = config.get("local_epochs", self.local_epochs)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        total_loss = 0.0
        total_samples = len(self.train_records)

        for epoch in range(epochs):
            for images, masks, _ in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                logits = self.model(images)
                loss, _ = self.criterion(logits, masks)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)

        mean_loss = total_loss / (max(total_samples, 1) * epochs)
        updated_parameters = get_parameters(self.model)

        return updated_parameters, total_samples, {
            "client_id": self.client_id,
            "train_loss": float(mean_loss)
        }

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluates current global model on hospital's local held-out test split.
        """
        set_parameters(self.model, parameters)
        self.model.eval()

        total_loss = 0.0
        total_samples = len(self.test_records)
        metrics_accum = {
            "liver_dice": 0.0, "tumor_dice": 0.0, "mean_dice": 0.0,
            "liver_iou": 0.0, "tumor_iou": 0.0,
            "liver_precision": 0.0, "tumor_precision": 0.0,
            "liver_recall": 0.0, "tumor_recall": 0.0
        }

        with torch.no_grad():
            for images, masks, _ in self.test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss, _ = self.criterion(logits, masks)
                total_loss += loss.item() * images.size(0)

                b_metrics = evaluate_batch(logits, masks)
                for k in metrics_accum:
                    metrics_accum[k] += b_metrics[k] * images.size(0)

        mean_loss = float(total_loss / max(total_samples, 1))
        client_eval = {k: float(v / max(total_samples, 1)) for k, v in metrics_accum.items()}
        client_eval["client_id"] = self.client_id

        return mean_loss, total_samples, client_eval
