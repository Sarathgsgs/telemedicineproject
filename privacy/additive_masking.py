"""
privacy/additive_masking.py
---------------------------
Simplified Zero-Sum Additive Masking Protocol for multi-client federated aggregation.
Each pair of hospital clients (i, j) shares a symmetric pairwise random mask:
Client i adds +M_{i,j}, Client j adds -M_{i,j}.
When the central server sums all masked updates, sum_{i} M_{i,j} = 0,
canceling all masks perfectly while hiding individual hospital weights from the server.
"""

import numpy as np
from typing import List, Dict, Tuple, Any


class AdditiveMaskingManager:
    """
    Manages pairwise mask generation and cancellation across K clients.
    """
    def __init__(self, num_clients: int, layer_shapes: List[Tuple[int, ...]], seed: int = 42):
        self.num_clients = num_clients
        self.layer_shapes = layer_shapes
        self.seed = seed
        self.pairwise_masks: Dict[Tuple[int, int], List[np.ndarray]] = {}
        self._generate_pairwise_masks()

    def _generate_pairwise_masks(self):
        """
        Generates zero-mean symmetric random masks for every client pair (i, j) with i < j.
        """
        rng = np.random.RandomState(self.seed)
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                pair_mask = []
                for shape in self.layer_shapes:
                    # Random Gaussian mask
                    mask = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
                    pair_mask.append(mask)
                self.pairwise_masks[(i, j)] = pair_mask

    def mask_client_parameters(
        self,
        client_id: int,
        parameters: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Applies symmetric pairwise masks to client_id's parameters:
        w_masked = w + sum_{j > i} M_{i,j} - sum_{j < i} M_{j,i}
        """
        masked_params = [p.copy() for p in parameters]

        for j in range(self.num_clients):
            if j == client_id:
                continue
            if j > client_id:
                # Add mask
                pair_mask = self.pairwise_masks[(client_id, j)]
                for l_idx in range(len(masked_params)):
                    masked_params[l_idx] += pair_mask[l_idx]
            else:
                # Subtract mask
                pair_mask = self.pairwise_masks[(j, client_id)]
                for l_idx in range(len(masked_params)):
                    masked_params[l_idx] -= pair_mask[l_idx]

        return masked_params

    def aggregate_masked_parameters(
        self,
        masked_client_parameters: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Server-side summation: sum_i w_masked_i = sum_i w_i + 0.
        All pairwise masks cancel out exactly.
        """
        num_layers = len(self.layer_shapes)
        unmasked_sum = []

        for l_idx in range(num_layers):
            layer_sum = sum(client_params[l_idx] for client_params in masked_client_parameters)
            unmasked_sum.append(layer_sum)

        return unmasked_sum
