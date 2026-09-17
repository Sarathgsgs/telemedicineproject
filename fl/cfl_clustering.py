"""
fl/cfl_clustering.py
--------------------
Flagship Contribution: Clustered Federated Learning (CFL-Lite) with Local Adaptation.
Computes cosine similarity between local client parameter update vectors (Delta w_i)
to dynamically group clients with congruent loss landscapes into clinical clusters,
mitigating non-IID gradient conflict and reducing cross-institutional variance.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.cluster import AgglomerativeClustering


def flatten_parameter_updates(
    current_params: List[np.ndarray],
    initial_params: List[np.ndarray]
) -> np.ndarray:
    """
    Computes and flattens the parameter update vector: Delta w = w_current - w_initial.
    """
    delta_list = []
    for cur, init in zip(current_params, initial_params):
        delta = (cur - init).ravel()
        delta_list.append(delta)
    return np.concatenate(delta_list)


def compute_cosine_similarity_matrix(update_vectors: List[np.ndarray]) -> np.ndarray:
    """
    Computes pairwise cosine similarity between client update vectors:
    S_{i,j} = <v_i, v_j> / (||v_i||_2 * ||v_j||_2)
    """
    num_clients = len(update_vectors)
    sim_matrix = np.zeros((num_clients, num_clients), dtype=np.float32)

    # Normalize vectors
    norms = []
    for v in update_vectors:
        norm = np.linalg.norm(v)
        norms.append(norm if norm > 1e-12 else 1e-12)

    for i in range(num_clients):
        for j in range(num_clients):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i < j:
                dot = np.dot(update_vectors[i], update_vectors[j])
                cos_sim = float(dot / (norms[i] * norms[j]))
                cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
                sim_matrix[i, j] = cos_sim
                sim_matrix[j, i] = cos_sim

    return sim_matrix


def cluster_clients_by_similarity(
    sim_matrix: np.ndarray,
    n_clusters: int = 2
) -> Dict[int, List[int]]:
    """
    Clusters clients into groups using Agglomerative Clustering on cosine distance (1 - S).
    Returns mapping: cluster_id -> list of client_ids
    """
    num_clients = sim_matrix.shape[0]
    if num_clients <= n_clusters:
        # Trivial cluster assignment if client count is small
        return {i: [i] for i in range(num_clients)}

    # Cosine distance = 1.0 - cosine similarity (bounded in [0, 2])
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(dist_matrix)

    clusters: Dict[int, List[int]] = {}
    for client_id, cluster_id in enumerate(labels):
        cluster_id = int(cluster_id)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(client_id)

    return clusters


def aggregate_within_clusters(
    cluster_mapping: Dict[int, List[int]],
    client_updates: List[Tuple[List[np.ndarray], int]]
) -> Dict[int, List[np.ndarray]]:
    """
    Performs dataset-size weighted parameter aggregation within each cluster independently.
    Returns: cluster_id -> aggregated parameter list
    """
    cluster_params: Dict[int, List[np.ndarray]] = {}

    for cluster_id, member_cids in cluster_mapping.items():
        total_cluster_samples = sum(client_updates[cid][1] for cid in member_cids)
        num_layers = len(client_updates[0][0])

        agg_layers = []
        for l_idx in range(num_layers):
            layer_sum = sum(
                (client_updates[cid][1] / total_cluster_samples) * client_updates[cid][0][l_idx]
                for cid in member_cids
            )
            agg_layers.append(layer_sum)

        cluster_params[cluster_id] = agg_layers

    return cluster_params
