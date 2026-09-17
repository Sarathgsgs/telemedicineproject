"""
tests/test_cfl.py
-----------------
Automated unit tests for Phase 5:
- Flattening parameter difference vectors (Delta w)
- Cosine similarity matrix properties (identity, symmetry, orthogonality)
- Agglomerative clustering by cosine distance
- Cluster-wise weighted aggregation correctness
"""

import unittest
import numpy as np

from fl.cfl_clustering import (
    flatten_parameter_updates,
    compute_cosine_similarity_matrix,
    cluster_clients_by_similarity,
    aggregate_within_clusters
)


class TestCFLClustering(unittest.TestCase):

    def test_flatten_parameter_updates(self):
        """Verifies Delta w calculation and concatenation."""
        init = [np.array([1.0, 2.0]), np.array([[3.0, 4.0], [5.0, 6.0]])]
        curr = [np.array([1.5, 2.5]), np.array([[3.5, 4.5], [5.5, 6.5]])]

        delta = flatten_parameter_updates(curr, init)
        expected = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual(delta.shape, (6,))
        self.assertTrue(np.allclose(delta, expected))

    def test_cosine_similarity_matrix(self):
        """Verifies similarity matrix properties: diagonal=1, symmetry, orthogonal=0, collinear=1."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])  # Collinear with v1
        v3 = np.array([0.0, 1.0, 0.0])  # Orthogonal to v1

        sim = compute_cosine_similarity_matrix([v1, v2, v3])

        # Diagonal must be 1.0
        self.assertAlmostEqual(sim[0, 0], 1.0)
        self.assertAlmostEqual(sim[1, 1], 1.0)
        self.assertAlmostEqual(sim[2, 2], 1.0)

        # Collinear vectors v1 and v2 have similarity 1.0
        self.assertAlmostEqual(sim[0, 1], 1.0, places=5)
        self.assertAlmostEqual(sim[1, 0], 1.0, places=5)

        # Orthogonal vectors v1 and v3 have similarity 0.0
        self.assertAlmostEqual(sim[0, 2], 0.0, places=5)
        self.assertAlmostEqual(sim[2, 0], 0.0, places=5)

    def test_client_clustering_assignments(self):
        """Verifies that collinear clients group into the same cluster."""
        # 3 clients: Client 0 & 1 have similar positive gradient alignment; Client 2 is divergent
        v0 = np.array([1.0, 1.0, 0.0])
        v1 = np.array([0.9, 1.1, 0.0])
        v2 = np.array([-1.0, -1.0, 5.0])

        sim = compute_cosine_similarity_matrix([v0, v1, v2])
        clusters = cluster_clients_by_similarity(sim, n_clusters=2)

        # 0 and 1 must be in the same cluster
        c0_cluster = next(c for c, members in clusters.items() if 0 in members)
        c1_cluster = next(c for c, members in clusters.items() if 1 in members)
        c2_cluster = next(c for c, members in clusters.items() if 2 in members)

        self.assertEqual(c0_cluster, c1_cluster)
        self.assertNotEqual(c0_cluster, c2_cluster)

    def test_aggregate_within_clusters(self):
        """Verifies cluster-wise weighted parameter averaging."""
        # Cluster 0: Client 0 (100 samples, w=2.0) and Client 1 (100 samples, w=4.0) -> Expected w = 3.0
        # Cluster 1: Client 2 (50 samples, w=10.0) -> Expected w = 10.0
        mapping = {0: [0, 1], 1: [2]}
        client_updates = [
            ([np.array([2.0, 2.0])], 100),
            ([np.array([4.0, 4.0])], 100),
            ([np.array([10.0, 10.0])], 50)
        ]

        agg = aggregate_within_clusters(mapping, client_updates)

        self.assertTrue(np.allclose(agg[0][0], np.array([3.0, 3.0])))
        self.assertTrue(np.allclose(agg[1][0], np.array([10.0, 10.0])))


if __name__ == "__main__":
    unittest.main()
