"""
data/non_iid_splitter.py
------------------------
Partitions CT slices across 3 to 5 simulated hospital clients with controlled clinical skew.
Enables direct comparison between IID baseline vs Non-IID skew, and sets up
the flagship Clustered Federated Learning (CFL-lite) personalization experiment.
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Any, Tuple


HOSPITAL_PROFILES = {
    0: {
        "name": "Metropolitan Cancer Center",
        "description": "Tertiary oncology center specializing in advanced, large hepatocellular carcinomas",
        "skew_profile": "large_skew",
        "target_bias": "large_tumor"
    },
    1: {
        "name": "Regional Diagnostic Clinic",
        "description": "Outpatient imaging center detecting early-stage, subtle focal hepatic lesions",
        "skew_profile": "small_skew",
        "target_bias": "small_tumor"
    },
    2: {
        "name": "Community Health Screening Center",
        "description": "Primary healthcare screening facility where >75% of patients have healthy liver scans",
        "skew_profile": "normal_skew",
        "target_bias": "normal"
    },
    3: {
        "name": "General District Hospital",
        "description": "Balanced clinical intake across mixed liver presentations",
        "skew_profile": "balanced",
        "target_bias": "balanced"
    }
}


def partition_slices_non_iid(
    records: List[Dict[str, Any]],
    num_clients: int = 3,
    seed: int = 42
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Partitions slice records into non-IID client sets based on clinical profiles.
    
    Category buckets:
      - 'large_tumor'
      - 'medium_tumor'
      - 'small_tumor'
      - 'normal'
    """
    random.seed(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "large_tumor": [],
        "medium_tumor": [],
        "small_tumor": [],
        "normal": []
    }

    for r in records:
        cat = r.get("category", "normal")
        if cat in buckets:
            buckets[cat].append(r)
        else:
            buckets["normal"].append(r)

    # Shuffle each bucket
    for k in buckets:
        random.shuffle(buckets[k])

    client_data: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(num_clients)}

    if num_clients == 3:
        # Hospital 0: Metropolitan Cancer Center (Large Tumors)
        # Allocate 70% of large, 20% of medium, 10% of small, 10% of normal
        p0_large = int(len(buckets["large_tumor"]) * 0.70)
        p0_med = int(len(buckets["medium_tumor"]) * 0.20)
        p0_small = int(len(buckets["small_tumor"]) * 0.10)
        p0_norm = int(len(buckets["normal"]) * 0.10)
        
        client_data[0].extend(buckets["large_tumor"][:p0_large])
        client_data[0].extend(buckets["medium_tumor"][:p0_med])
        client_data[0].extend(buckets["small_tumor"][:p0_small])
        client_data[0].extend(buckets["normal"][:p0_norm])

        # Hospital 1: Regional Diagnostic Clinic (Small Tumors)
        # Allocate 10% of large, 40% of medium, 70% of small, 15% of normal
        p1_large = int(len(buckets["large_tumor"]) * 0.15)
        p1_med = int(len(buckets["medium_tumor"]) * 0.40)
        p1_small = int(len(buckets["small_tumor"]) * 0.70)
        p1_norm = int(len(buckets["normal"]) * 0.15)

        client_data[1].extend(buckets["large_tumor"][p0_large : p0_large + p1_large])
        client_data[1].extend(buckets["medium_tumor"][p0_med : p0_med + p1_med])
        client_data[1].extend(buckets["small_tumor"][p0_small : p0_small + p1_small])
        client_data[1].extend(buckets["normal"][p0_norm : p0_norm + p1_norm])

        # Hospital 2: Community Screening (Healthy Heavy)
        # Receives the remainder (dominated by healthy/normal scans)
        client_data[2].extend(buckets["large_tumor"][p0_large + p1_large:])
        client_data[2].extend(buckets["medium_tumor"][p0_med + p1_med:])
        client_data[2].extend(buckets["small_tumor"][p0_small + p1_small:])
        client_data[2].extend(buckets["normal"][p0_norm + p1_norm:])

    else:
        # Generic Dirichlet partitioning for N clients
        alpha = 0.5  # Skew parameter
        for cat, items in buckets.items():
            if len(items) == 0:
                continue
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            counts = (proportions * len(items)).astype(int)
            counts[-1] = len(items) - counts[:-1].sum()
            
            start = 0
            for i, c in enumerate(counts):
                client_data[i].extend(items[start : start + c])
                start += c

    return client_data


def partition_slices_iid(
    records: List[Dict[str, Any]],
    num_clients: int = 3,
    seed: int = 42
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Partitions slice records uniformly at random (IID baseline).
    """
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    
    splits = np.array_split(shuffled, num_clients)
    return {i: list(s) for i, s in enumerate(splits)}


def summarize_partition(client_data: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Computes summary distribution metrics across all clients.
    """
    summary = {}
    for cid, samples in client_data.items():
        categories = {}
        total_tumor_pixels = 0
        total_liver_pixels = 0

        for s in samples:
            cat = s.get("category", "normal")
            categories[cat] = categories.get(cat, 0) + 1
            total_tumor_pixels += s.get("tumor_pixels", 0)
            total_liver_pixels += s.get("liver_pixels", 1)

        mean_ratio = float(total_tumor_pixels / total_liver_pixels) if total_liver_pixels > 0 else 0.0
        
        info = HOSPITAL_PROFILES.get(cid, {"name": f"Hospital Node {cid}", "target_bias": "custom"})
        summary[f"client_{cid}"] = {
            "hospital_name": info["name"],
            "target_bias": info["target_bias"],
            "total_slices": len(samples),
            "category_breakdown": categories,
            "mean_tumor_burden": round(mean_ratio * 100, 2)  # In percentage %
        }
    return summary
