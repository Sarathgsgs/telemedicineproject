"""
data/init_dataset.py
--------------------
Orchestration script for Phase 2 data pipeline.
1. Generates synthetic cohort (or extracts from LiTS NIfTI files if available).
2. Performs controlled Non-IID clinical partitioning across 3 hospital institutions.
3. Generates a multi-panel visual sanity check plot:
   [Raw HU Slice] | [Ground Truth Overlay] | [Augmented Slice]
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import argparse
from typing import Optional, List, Dict, Any, Tuple
from data.synthetic_generator import generate_synthetic_cohort, create_synthetic_ct_slice
from data.non_iid_splitter import partition_slices_non_iid, partition_slices_iid, summarize_partition
from data.dataset import LiverSliceDataset, get_dataloader
from data.patient_splitter import partition_by_patient, save_patient_partitions


def setup_dataset(
    data_root: str = "data/processed",
    total_samples: int = 150,
    num_clients: int = 3,
    mode: str = "synthetic",
    raw_lits_dir: str = "data/dataset",
    max_volumes: Optional[int] = None
):
    print("=" * 65)
    print(f"DATA PIPELINE INITIALIZATION & PARTITIONING (Mode: {mode.upper()})")
    print("=" * 65)

    if mode == "real_lits":
        from data.lits_downloader import check_local_lits_files, ingest_and_extract_lits
        pairs = check_local_lits_files(raw_lits_dir)
        if not pairs:
            print(f"[-] No matched LiTS NIfTI pairs found in {raw_lits_dir}.")
            from data.lits_downloader import print_download_instructions
            print_download_instructions()
            return None, None

        real_cohort_dir = os.path.join(data_root, "real_lits", "slices")
        records = ingest_and_extract_lits(
            raw_dir=raw_lits_dir,
            output_dir=real_cohort_dir,
            max_volumes=max_volumes
        )
        
        # Patient-Level Partitioning (Eliminates Data Leakage)
        print("\n[*] Enforcing strict patient-level partition across train, val, and test...")
        partition_data = partition_by_patient(records, num_clients=num_clients)
        
        manifest_file = os.path.join(data_root, "real_lits", "client_partitions_patient_level.json")
        save_patient_partitions(partition_data, manifest_file)
        print(f"[+] Real LiTS patient-level manifest saved: {manifest_file}")

        # Render sanity check plot for real LiTS slices
        plot_path = os.path.join(data_root, "real_lits", "sanity_check_real_lits.png")
        if len(records) >= 4:
            generate_sanity_plot(records[:4], plot_path)
            print(f"[+] Real CT visual verification plot saved to: {plot_path}")

        return manifest_file, plot_path

    # Default / Synthetic Mode
    os.makedirs(data_root, exist_ok=True)
    cohort_dir = os.path.join(data_root, "slices")
    os.makedirs(cohort_dir, exist_ok=True)

    # 1. Generate cohort
    print(f"[*] Generating {total_samples} clinical CT slices with HU windowing [-100, 400]...")
    records = generate_synthetic_cohort(
        output_dir=cohort_dir,
        num_samples=total_samples,
        skew_profile="balanced",
        client_id="cohort_master"
    )
    print(f"[+] Successfully generated {len(records)} 2D axial CT slices.")

    # 2. Partition into Non-IID Hospital Clients
    print(f"[*] Partitioning dataset across {num_clients} simulated hospital nodes with clinical skew...")
    non_iid_splits = partition_slices_non_iid(records, num_clients=num_clients)
    iid_splits = partition_slices_iid(records, num_clients=num_clients)


    non_iid_summary = summarize_partition(non_iid_splits)
    iid_summary = summarize_partition(iid_splits)

    print("\n--- Non-IID Institutional Distribution ---")
    for cid, info in non_iid_summary.items():
        print(f"  {info['hospital_name']} ({cid}):")
        print(f"    Target Bias: {info['target_bias']}")
        print(f"    Total Slices: {info['total_slices']}")
        print(f"    Categories: {info['category_breakdown']}")
        print(f"    Mean Tumor Burden: {info['mean_tumor_burden']}%\n")

    # Save partition manifests
    partitions_manifest = {
        "num_clients": num_clients,
        "total_samples": total_samples,
        "non_iid_summary": non_iid_summary,
        "iid_summary": iid_summary,
        "non_iid_splits": {str(k): v for k, v in non_iid_splits.items()},
        "iid_splits": {str(k): v for k, v in iid_splits.items()}
    }

    manifest_file = os.path.join(data_root, "client_partitions.json")
    with open(manifest_file, "w") as f:
        json.dump(partitions_manifest, f, indent=2)
    print(f"[+] Partitions saved to: {manifest_file}")

    # 3. Visual Sanity Check
    print("[*] Rendering visual sanity check plot: Raw Slice | Ground Truth Overlay | Augmented Slice...")
    plot_path = os.path.join(data_root, "sanity_check_samples.png")
    generate_sanity_plot(records[:4], plot_path)
    print(f"[+] Visual verification plot saved to: {plot_path}")
    print("=" * 65)
    return manifest_file, plot_path


def generate_sanity_plot(sample_records, output_path: str):
    """
    Renders side-by-side comparison for 4 samples:
    Column 1: Raw CT Slice (HU windowed)
    Column 2: Ground Truth Overlay (Green = Liver Parenchyma, Red = Tumor)
    Column 3: Augmented Slice (Rotation, Scaling, Noise)
    """
    dataset = LiverSliceDataset(sample_records, augment=True)
    num_samples = len(sample_records)
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3.2 * num_samples))

    for i in range(num_samples):
        # Unaugmented
        with np.load(sample_records[i]["filepath"]) as d:
            raw_img = d["image"]
            mask = d["mask"]

        # Augmented from dataset
        aug_tensor, aug_mask_tensor, _ = dataset[i]
        aug_img = aug_tensor.squeeze(0).numpy()
        aug_mask = aug_mask_tensor.numpy()

        # RGB overlay
        overlay = np.stack([raw_img, raw_img, raw_img], axis=-1)
        # Liver (Class 1) in semi-transparent green
        liver_idx = mask == 1
        overlay[liver_idx, 0] = overlay[liver_idx, 0] * 0.4
        overlay[liver_idx, 1] = np.clip(overlay[liver_idx, 1] * 0.4 + 0.6, 0, 1)
        overlay[liver_idx, 2] = overlay[liver_idx, 2] * 0.4
        # Tumor (Class 2) in bright red
        tumor_idx = mask == 2
        overlay[tumor_idx, 0] = 1.0
        overlay[tumor_idx, 1] = 0.15
        overlay[tumor_idx, 2] = 0.15

        cat = sample_records[i].get("category", "sample")

        # Plot Raw
        axes[i, 0].imshow(raw_img, cmap="gray")
        axes[i, 0].set_title(f"Raw CT Slice (HU Windowed)\n[{cat}]", fontsize=10)
        axes[i, 0].axis("off")

        # Plot Overlay
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title("Ground Truth Overlay\n(Green: Liver | Red: Tumor)", fontsize=10)
        axes[i, 1].axis("off")

        # Plot Augmented
        axes[i, 2].imshow(aug_img, cmap="gray")
        axes[i, 2].set_title("Augmented Slice\n(Rotation/Scale/Noise)", fontsize=10)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# Alias for launcher compatibility
init_dataset = setup_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedLiverNet Data Pipeline Initializer")
    default_mode = "real_lits" if os.path.exists("data/dataset") else "synthetic"
    parser.add_argument("--mode", choices=["synthetic", "real_lits"], default=default_mode,
                        help="Data mode: 'synthetic' or 'real_lits'")
    parser.add_argument("--raw-dir", type=str, default="data/dataset",
                        help="Directory containing real LiTS NIfTI files")
    parser.add_argument("--max-volumes", type=int, default=10,
                        help="Maximum patient volumes to extract for real LiTS")
    parser.add_argument("--samples", type=int, default=150,
                        help="Total synthetic samples (if mode=synthetic)")
    parser.add_argument("--clients", type=int, default=3,
                        help="Number of simulated hospital clients")
    args = parser.parse_args()

    setup_dataset(
        total_samples=args.samples,
        num_clients=args.clients,
        mode=args.mode,
        raw_lits_dir=args.raw_dir,
        max_volumes=args.max_volumes
    )
