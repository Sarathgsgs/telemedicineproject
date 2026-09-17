"""
data/synthetic_generator.py
---------------------------
High-fidelity synthetic abdominal CT slice generator for liver lesion screening.
Generates 2D axial CT slices with anatomically plausible liver lobes, vessels,
and tumor lesions adhering to clinical Hounsfield Unit distributions and noise physics.
Enables instant local development, verification, and UI demonstrations without 50GB downloads.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy.ndimage import gaussian_filter

HU_MIN = -100.0
HU_MAX = 400.0


def create_synthetic_ct_slice(
    size: int = 256,
    tumor_type: str = "random",
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Synthesizes a realistic 2D axial abdominal CT slice and paired segmentation mask.
    
    Args:
        size: Image height and width (default: 256).
        tumor_type: 'none', 'small', 'medium', 'large', or 'random'.
        seed: Random seed for reproducibility.
        
    Returns:
        (image_normalized, mask, metadata)
        - image_normalized: float32 array in [0, 1] (clipped to [-100, 400] HU)
        - mask: uint8 array with classes {0: background, 1: liver, 2: tumor}
        - metadata: clinical lesion metrics dictionary
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 1. Coordinate Grid
    y, x = np.mgrid[0:size, 0:size]
    cy, cx = size / 2.0, size / 2.0

    # Start with air HU background (-1000 HU)
    ct_hu = np.full((size, size), -1000.0, dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    # 2. Body Envelope (Elliptical abdomen)
    body_rx, body_ry = size * 0.43, size * 0.36
    body_mask = (((x - cx) / body_rx)**2 + ((y - cy) / body_ry)**2) <= 1.0
    ct_hu[body_mask] = -60.0  # Subcutaneous fat & generic soft tissue (-60 HU to 20 HU)

    # 3. Spine & Ribs (Bone ~ 300 to 500 HU)
    spine_mask = (((x - cx) / (size * 0.07))**2 + ((y - (cy + size * 0.22)) / (size * 0.08))**2) <= 1.0
    ct_hu[spine_mask] = 380.0

    # 4. Liver Lobe (Anatomical Right - left side of image, anterior-lateral)
    liver_cx, liver_cy = cx - size * 0.16, cy - size * 0.04
    # Create organic, asymmetric liver shape using combined ellipses + deformation
    angle = np.radians(25)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lx = (x - liver_cx) * cos_a - (y - liver_cy) * sin_a
    ly = (x - liver_cx) * sin_a + (y - liver_cy) * cos_a

    # Asymmetric lobes
    liver_mask_base = ((lx / (size * 0.26))**2 + (ly / (size * 0.23))**2) <= 1.0
    # Indentation for gall bladder / inferior vena cava
    indent = (((x - (liver_cx + size * 0.12)) / (size * 0.09))**2 + ((y - (liver_cy + size * 0.08)) / (size * 0.08))**2) <= 1.0
    liver_mask = liver_mask_base & (~indent) & body_mask

    # Liver Parenchyma CT Attenuation: ~60-80 HU
    ct_hu[liver_mask] = np.random.normal(68.0, 6.0, size=np.sum(liver_mask))
    mask[liver_mask] = 1

    # 5. Intrahepatic Portal Vessels (slightly hypodense or hyperdense)
    vessel_core = (((x - (liver_cx + size * 0.04)) / (size * 0.03))**2 + ((y - (liver_cy - size * 0.02)) / (size * 0.03))**2) <= 1.0
    vessel_mask = vessel_core & liver_mask
    ct_hu[vessel_mask] = 110.0

    # 6. Tumors / Lesions
    if tumor_type == "random":
        tumor_type = random.choices(["none", "small", "medium", "large"], weights=[0.25, 0.30, 0.25, 0.20])[0]

    tumor_pixels = 0
    if tumor_type != "none" and np.sum(liver_mask) > 500:
        # Determine tumor radius based on category
        if tumor_type == "small":
            t_radius = random.uniform(size * 0.025, size * 0.045)  # ~6-11 px
        elif tumor_type == "medium":
            t_radius = random.uniform(size * 0.050, size * 0.080)  # ~12-20 px
        else:  # large
            t_radius = random.uniform(size * 0.085, size * 0.130)  # ~21-33 px

        # Find valid center inside liver
        liver_indices = np.argwhere(liver_mask)
        # Avoid outer border by sampling near middle quartile of liver points
        center_candidates = liver_indices[len(liver_indices) // 4 : 3 * len(liver_indices) // 4]
        if len(center_candidates) > 0:
            tc_y, tc_x = center_candidates[random.randint(0, len(center_candidates) - 1)]
            
            # Deformed lesion boundary
            dist = np.sqrt((x - tc_x)**2 + (y - tc_y)**2)
            # Add subtle boundary irregularity
            perturbation = np.random.normal(0, t_radius * 0.12, size=(size, size))
            perturbation = gaussian_filter(perturbation, sigma=1.5)
            tumor_candidate = (dist + perturbation) <= t_radius
            
            # Constrain tumor strictly within liver parenchyma
            tumor_mask = tumor_candidate & liver_mask
            tumor_pixels = int(np.sum(tumor_mask))

            if tumor_pixels > 0:
                mask[tumor_mask] = 2
                # Lesions are typically hypodense relative to liver (30-45 HU) with central necrosis for large tumors
                if tumor_type == "large":
                    ct_hu[tumor_mask] = np.random.normal(32.0, 7.0, size=tumor_pixels)
                    # Necrotic core
                    necrotic = dist <= (t_radius * 0.4)
                    necrotic_mask = necrotic & tumor_mask
                    ct_hu[necrotic_mask] = np.random.normal(15.0, 4.0, size=np.sum(necrotic_mask))
                else:
                    ct_hu[tumor_mask] = np.random.normal(38.0, 5.0, size=tumor_pixels)

    # 7. Add Simulated CT Quantum Mottle Noise
    quantum_noise = np.random.normal(0.0, 7.5, size=(size, size))
    ct_hu += quantum_noise

    # 8. Apply Clinical HU Windowing [-100, 400] and Normalization to [0, 1]
    clipped = np.clip(ct_hu, HU_MIN, HU_MAX)
    image_normalized = (clipped - HU_MIN) / (HU_MAX - HU_MIN)
    image_normalized = image_normalized.astype(np.float32)

    liver_pixels = int(np.sum(mask >= 1))
    metadata = {
        "liver_pixels": liver_pixels,
        "tumor_pixels": tumor_pixels,
        "tumor_ratio": round(float(tumor_pixels / liver_pixels), 4) if liver_pixels > 0 else 0.0,
        "has_tumor": bool(tumor_pixels > 0),
        "category": "normal" if tumor_pixels == 0 else f"{tumor_type}_tumor"
    }

    return image_normalized, mask, metadata


def generate_synthetic_cohort(
    output_dir: str,
    num_samples: int = 120,
    skew_profile: str = "balanced",
    client_id: str = "client_global"
) -> List[Dict[str, Any]]:
    """
    Generates a synthetic cohort of axial CT slices and saves them as compressed .npz archives.
    
    Skew profiles for Controlled Non-IID simulation:
      - 'large_skew': 65% large tumors, 20% medium, 10% small, 5% normal (Hospital A - Cancer Center)
      - 'small_skew': 10% large, 25% medium, 55% small, 10% normal (Hospital B - Diagnostic Clinic)
      - 'normal_skew': 5% large, 10% medium, 10% small, 75% normal (Hospital C - Community Screening)
      - 'balanced': Equal distribution across classes
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    if skew_profile == "large_skew":
        weights = [0.05, 0.10, 0.20, 0.65]  # [none, small, medium, large]
    elif skew_profile == "small_skew":
        weights = [0.10, 0.55, 0.25, 0.10]
    elif skew_profile == "normal_skew":
        weights = [0.75, 0.10, 0.10, 0.05]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]

    type_choices = ["none", "small", "medium", "large"]

    for idx in range(num_samples):
        chosen_type = random.choices(type_choices, weights=weights)[0]
        img, mask, meta = create_synthetic_ct_slice(
            size=256,
            tumor_type=chosen_type,
            seed=42 + idx * 7
        )

        filename = f"{client_id}_slice_{idx:04d}.npz"
        filepath = os.path.join(output_dir, filename)

        np.savez_compressed(filepath, image=img, mask=mask)

        record = {
            "client_id": client_id,
            "filename": filename,
            "filepath": filepath,
            "skew_profile": skew_profile,
            **meta
        }
        manifest.append(record)

    manifest_path = os.path.join(output_dir, f"{client_id}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
