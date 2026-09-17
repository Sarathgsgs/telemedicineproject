"""
comm/topk_sparsify.py
---------------------
Top-k Gradient / Parameter Sparsification with Local Error-Feedback Buffer (EF-SGD).
Transmits only the top p% largest magnitude updates while accumulating residuals
in a local error buffer, guaranteeing zero gradient loss and significant bandwidth reduction.
Replicates FedLiverNet Table 7 communication-efficiency mechanisms.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional


class TopKSparsifier:
    """
    Manages Top-k parameter compression and local error-feedback buffers.
    """
    def __init__(self, density_ratio: float = 0.10):
        """
        Args:
            density_ratio: Fraction of parameters to transmit (e.g. 0.10 = top 10% sent, 90% sparse).
        """
        self.density_ratio = max(0.01, min(1.0, density_ratio))
        # Client ID -> list of error buffer arrays matching model layer shapes
        self.error_buffers: Dict[int, List[np.ndarray]] = {}

    def init_client_buffer(self, client_id: int, layer_shapes: List[Tuple[int, ...]]):
        """Initializes zero-filled error accumulation buffer for a hospital client."""
        self.error_buffers[client_id] = [np.zeros(s, dtype=np.float32) for s in layer_shapes]

    def compress_updates(
        self,
        client_id: int,
        updates: List[np.ndarray]
    ) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
        """
        Compresses client updates with Error Feedback:
        v_t = Delta w_t + e_{t-1}
        sparse_v = topk(v_t)
        e_t = v_t - sparse_v
        
        Returns:
            (sparse_layers, metadata)
        """
        # Ensure error buffer exists
        if client_id not in self.error_buffers:
            self.init_client_buffer(client_id, [u.shape for u in updates])

        sparse_payload = []
        total_dense_params = 0
        total_sparse_params = 0

        for l_idx, layer in enumerate(updates):
            # 1. Add accumulated error from previous round
            v_t = layer + self.error_buffers[client_id][l_idx]
            flat_v = v_t.flatten()
            num_params = len(flat_v)
            total_dense_params += num_params

            if self.density_ratio >= 1.0:
                # Dense bypass (no sparsification)
                sparse_payload.append({
                    "is_dense": True,
                    "shape": layer.shape,
                    "values": v_t
                })
                self.error_buffers[client_id][l_idx].fill(0.0)
                total_sparse_params += num_params
                continue

            # 2. Determine top-k threshold by absolute magnitude
            k = max(1, int(np.ceil(self.density_ratio * num_params)))
            total_sparse_params += k

            # Partition indices for top-k absolute values
            abs_v = np.abs(flat_v)
            topk_indices = np.argpartition(abs_v, -k)[-k:]
            topk_values = flat_v[topk_indices]

            # 3. Update error feedback buffer: e_t = v_t - sparse_v
            # Clear transmitted positions in error buffer
            flat_error = flat_v.copy()
            flat_error[topk_indices] = 0.0
            self.error_buffers[client_id][l_idx] = flat_error.reshape(layer.shape)

            sparse_payload.append({
                "is_dense": False,
                "shape": layer.shape,
                "indices": topk_indices.astype(np.int32),
                "values": topk_values.astype(np.float32)
            })

        # Calculate exact payload size in bytes and MB
        dense_bytes = total_dense_params * 4  # float32
        # Sparse format: value (4 bytes) + int32 index (4 bytes) = 8 bytes per sparse entry
        sparse_bytes = total_sparse_params * 8 if self.density_ratio < 1.0 else dense_bytes

        dense_mb = float(dense_bytes / (1024 * 1024))
        sparse_mb = float(sparse_bytes / (1024 * 1024))
        savings_pct = round((1.0 - sparse_bytes / max(dense_bytes, 1)) * 100, 2)

        meta = {
            "client_id": client_id,
            "density_ratio": self.density_ratio,
            "sparsity_pct": round((1.0 - self.density_ratio) * 100, 1),
            "total_parameters": total_dense_params,
            "transmitted_parameters": total_sparse_params,
            "dense_mb": round(dense_mb, 3),
            "sparse_mb": round(sparse_mb, 3),
            "bandwidth_savings_pct": savings_pct
        }

        return sparse_payload, meta

    def decompress_updates(
        self,
        sparse_payload: List[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Reconstructs full parameter update layers on the aggregation server.
        """
        reconstructed = []
        for item in sparse_payload:
            shape = item["shape"]
            if item.get("is_dense", False):
                reconstructed.append(item["values"])
            else:
                flat_rec = np.zeros(np.prod(shape), dtype=np.float32)
                indices = item["indices"]
                values = item["values"]
                flat_rec[indices] = values
                reconstructed.append(flat_rec.reshape(shape))
        return reconstructed
