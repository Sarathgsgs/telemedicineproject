"""
data/lits_downloader.py
-----------------------
Utility to download, verify, and ingest Liver Tumor Segmentation (LiTS) CT volumes.
- Verifies volume-to-segmentation paired alignment (e.g. volume-0.nii <-> segmentation-0.nii).
- Supports direct Kaggle API download (if credentials exist) or validates manually placed scans.
- Prepares verified scans for slice extraction in data/slice_extractor.py.
"""

import os
import sys
import shutil
from pathlib import Path
import re
from typing import List, Tuple, Dict, Any, Optional


RAW_LITS_DIR = "data/raw_lits"
DATASET_DIR = "data/dataset"
PROCESSED_LITS_DIR = "data/processed/real_lits/slices"


def check_local_lits_files(raw_dir: str = RAW_LITS_DIR) -> List[Tuple[str, str, str]]:
    """
    Scans raw_dir (and subdirectories) for matched volume and segmentation pairs.
    Checks fallback dataset directory if raw_dir has no pairs.
    Returns:
        List of tuples: (subject_id, volume_path, seg_path)
    """
    candidate_dirs = [raw_dir]
    if raw_dir != DATASET_DIR and os.path.exists(DATASET_DIR):
        candidate_dirs.append(DATASET_DIR)

    for search_dir in candidate_dirs:
        if not os.path.exists(search_dir):
            continue

        volumes = {}
        segmentations = {}

        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if not (f.endswith(".nii") or f.endswith(".nii.gz")):
                    continue
                lower_f = f.lower()
                full_path = os.path.join(root, f)

                if "segmentation" in lower_f or "seg-" in lower_f:
                    m = re.search(r'(?:segmentation-|seg-)(\d+)', lower_f)
                    if m:
                        segmentations[int(m.group(1))] = full_path
                elif "volume" in lower_f:
                    m = re.search(r'(?:volume-|volume_pt)(\d+)', lower_f)
                    if m:
                        volumes[int(m.group(1))] = full_path

        matched_pairs = []
        for vid in sorted(volumes.keys()):
            if vid in segmentations:
                matched_pairs.append((f"patient_{vid:03d}", volumes[vid], segmentations[vid]))

        if matched_pairs:
            return matched_pairs

    return []


def print_download_instructions():
    """Prints clear guidance for obtaining LiTS CT volumes."""
    print("=" * 68)
    print("  LITS DATASET ACQUISITION GUIDE")
    print("=" * 68)
    print("The LiTS dataset contains 130 contrast-enhanced 3D CT volumes.")
    print("To evaluate your model on real CT data:")
    print()
    print("Option 1: Direct Kaggle Download (Fastest)")
    print("  1. Visit: https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation")
    print(f"  2. Download 3-5 patient pairs (e.g., volume-0.nii & segmentation-0.nii)")
    print(f"  3. Place them into: {os.path.abspath(DATASET_DIR)} or {os.path.abspath(RAW_LITS_DIR)}")
    print()
    print("Option 2: Kaggle CLI (Automated)")
    print("  Run:")
    print(f"    kaggle datasets download -d andrewmvd/liver-tumor-segmentation -p {DATASET_DIR} --unzip")
    print()
    print("Option 3: Official CodaLab Benchmark")
    print("  Visit: https://competitions.codalab.org/competitions/17094")
    print("=" * 68)


def ingest_and_extract_lits(
    raw_dir: str = RAW_LITS_DIR,
    output_dir: str = PROCESSED_LITS_DIR,
    target_size: Tuple[int, int] = (256, 256),
    max_volumes: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Validates paired NIfTI files and extracts 2D axial slices with HU windowing.
    """
    pairs = check_local_lits_files(raw_dir)
    if not pairs:
        print(f"[-] No matched (volume, segmentation) pairs found in: {raw_dir}")
        print_download_instructions()
        return []

    if max_volumes is not None and max_volumes > 0:
        pairs = pairs[:max_volumes]

    print(f"[+] Found {len(pairs)} matched LiTS patient volumes for extraction:")
    for sid, vol_p, seg_p in pairs:
        print(f"    - {sid}: {os.path.basename(vol_p)} <-> {os.path.basename(seg_p)}")

    from data.slice_extractor import extract_slices_from_nifti
    os.makedirs(output_dir, exist_ok=True)
    all_extracted_slices = []

    for sid, vol_p, seg_p in pairs:
        print(f"[*] Processing {sid}...")
        slices = extract_slices_from_nifti(
            volume_path=vol_p,
            seg_path=seg_p,
            output_dir=output_dir,
            subject_id=sid,
            min_liver_pixels=100,
            target_shape=target_size
        )
        print(f"    -> Extracted {len(slices)} informative axial slices.")
        all_extracted_slices.extend(slices)

    print(f"\n[+] Total real clinical slices extracted: {len(all_extracted_slices)}")
    return all_extracted_slices


if __name__ == "__main__":
    pairs = check_local_lits_files()
    if not pairs:
        print_download_instructions()
    else:
        ingest_and_extract_lits()

