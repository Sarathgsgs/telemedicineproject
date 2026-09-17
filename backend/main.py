"""
backend/main.py
---------------
FastAPI Backend for the FedLiverNet Telemedicine Platform.
Integrates:
- EdgeInferenceEngine for zero-dependency real-time CPU segmentation.
- OfflineSyncManager for SQLite WAL offline queueing and reconnection sync replay.
- Teleconsultation endpoints: /api/predict, /api/predict_sample, /api/samples.
- Federated Learning Observatory: /api/fl_observatory (Tables 4, 5, 7 metrics & hospital nodes).
- Clinician feedback & audit logging.
- Static file serving for the frontend single-page application.
"""

import os
import io
import time
import json
import base64
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from edge.offline_manager import EdgeInferenceEngine, OfflineSyncManager, EdgePredictionResult

app = FastAPI(
    title="FedLiverNet Telemedicine Platform API",
    description="Privacy-Preserving Federated Learning Telemedicine System for Hepatic Lesion Screening",
    version="1.0.0"
)

# Enable CORS for local dev and cross-origin embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine and Offline Sync Manager
MODEL_PATH = "checkpoints/edge_model_traced.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "checkpoints/centralized_best.pt"

edge_engine = EdgeInferenceEngine(model_path=MODEL_PATH, device="cpu", base_filters=32)
sync_manager = OfflineSyncManager(db_path="edge/offline_store.db")

# Path to processed sample slices
SLICES_DIR = "data/processed/slices"
MANIFEST_PATH = os.path.join(SLICES_DIR, "cohort_master_manifest.json")


def array_to_base64_png(array_2d: np.ndarray) -> str:
    """Converts a normalized 2D numpy array [0, 1] to a grayscale base64 PNG data URI."""
    uint8_img = (np.clip(array_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def mask_to_rgba_base64_png(mask_2d: np.ndarray) -> str:
    """
    Converts a multi-class segmentation mask (0=bg, 1=liver, 2=tumor)
    into an RGBA colored transparent PNG overlay.
    - Liver: Emerald Green (16, 185, 129, 180)
    - Tumor: Vivid Amber-Red (239, 68, 68, 230)
    - Background: Transparent (0, 0, 0, 0)
    """
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Liver: Emerald #10B981
    liver_mask = (mask_2d == 1)
    rgba[liver_mask, 0] = 16
    rgba[liver_mask, 1] = 185
    rgba[liver_mask, 2] = 129
    rgba[liver_mask, 3] = 180

    # Tumor: Crimson #EF4444
    tumor_mask = (mask_2d == 2)
    rgba[tumor_mask, 0] = 239
    rgba[tumor_mask, 1] = 68
    rgba[tumor_mask, 2] = 68
    rgba[tumor_mask, 3] = 230

    pil_img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def get_curated_samples() -> List[Dict[str, Any]]:
    """Returns curated demonstration slices with clinical categories."""
    curated = [
        {
            "sample_id": "slice_0001",
            "title": "Clinical Case 1: Early-Stage Focal HCC Lesion",
            "patient_id": "PAT-LIT-104",
            "category": "Focal HCC Lesion",
            "clinical_notes": "Small suspicious hypodense focal nodule in right hepatic lobe.",
            "filename": "cohort_master_slice_0001.npz"
        },
        {
            "sample_id": "slice_0002",
            "title": "Clinical Case 2: Macro Hepatocellular Carcinoma",
            "patient_id": "PAT-LIT-078",
            "category": "Macro HCC Tumor",
            "clinical_notes": "Large hypervascular neoplasm infiltrating central hepatic parenchyma.",
            "filename": "cohort_master_slice_0002.npz"
        },
        {
            "sample_id": "slice_0000",
            "title": "Clinical Case 3: Moderate Hepatic Tumor Nodule",
            "patient_id": "PAT-LIT-055",
            "category": "Medium Lesion",
            "clinical_notes": "Circumscribed lesion with mild peritumoral edema.",
            "filename": "cohort_master_slice_0000.npz"
        },
        {
            "sample_id": "slice_healthy",
            "title": "Clinical Case 4: Healthy Liver Parenchyma (Control)",
            "patient_id": "PAT-LIT-012",
            "category": "Healthy Normal",
            "clinical_notes": "Uniform hepatic attenuation without discrete focal parenchymal lesions.",
            "filename": "cohort_master_slice_0015.npz"
        }
    ]
    return curated


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------

@app.get("/api/status")
def get_system_status():
    """Returns runtime health, edge node online/offline state, and queue stats."""
    q_status = sync_manager.get_queue_status()
    return {
        "status": "ONLINE",
        "model_architecture": "2D Attention U-Net-Lite (GroupNorm + Attention Gates)",
        "model_runtime": "TorchScript FP32 Edge Engine",
        "device": edge_engine.device,
        "edge_connectivity": q_status["network_status"],
        "active_fl_cluster": "CFL Cluster 1 (High-Prevalence Skew)",
        "offline_queue": q_status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/api/samples")
def get_samples():
    """Returns curated clinical cases available for one-click testing."""
    return {"samples": get_curated_samples()}


def standardize_slice_and_mask(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardizes CT slice and mask to (128, 128) resolution."""
    if img.shape != (128, 128):
        pil_img = Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
        pil_img = pil_img.resize((128, 128), resample=Image.BILINEAR)
        img = np.array(pil_img, dtype=np.float32) / 255.0

    if mask.shape != (128, 128):
        pil_mask = Image.fromarray(mask.astype(np.uint8))
        pil_mask = pil_mask.resize((128, 128), resample=Image.NEAREST)
        mask = np.array(pil_mask, dtype=np.uint8)

    return img, mask


@app.get("/api/sample/{sample_id}")
def get_sample_details(sample_id: str):
    """Retrieves raw image preview and ground-truth mask for a sample slice."""
    curated = {s["sample_id"]: s for s in get_curated_samples()}
    if sample_id not in curated:
        raise HTTPException(status_code=404, detail="Sample not found")

    sample_meta = curated[sample_id]
    filepath = os.path.join(SLICES_DIR, sample_meta["filename"])

    if not os.path.exists(filepath):
        # Generate synthetic fallback if processed slice missing
        img = np.random.uniform(0.1, 0.9, (128, 128)).astype(np.float32)
        gt_mask = np.zeros((128, 128), dtype=np.uint8)
    else:
        data = np.load(filepath)
        img = data["image"]
        gt_mask = data["mask"]

    img, gt_mask = standardize_slice_and_mask(img, gt_mask)

    return {
        "sample_meta": sample_meta,
        "raw_slice_b64": array_to_base64_png(img),
        "gt_mask_b64": mask_to_rgba_base64_png(gt_mask),
        "has_tumor": bool(np.any(gt_mask == 2)),
        "liver_pixels": int(np.sum(gt_mask == 1)),
        "tumor_pixels": int(np.sum(gt_mask == 2))
    }


class PredictSampleRequest(BaseModel):
    sample_id: str


@app.post("/api/predict_sample")
def predict_sample(req: PredictSampleRequest):
    """Executes edge segmentation inference on a selected clinical sample."""
    curated = {s["sample_id"]: s for s in get_curated_samples()}
    if req.sample_id not in curated:
        raise HTTPException(status_code=404, detail="Sample ID not found")

    meta = curated[req.sample_id]
    filepath = os.path.join(SLICES_DIR, meta["filename"])

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Slice file not found: {filepath}")

    data = np.load(filepath)
    raw_slice = data["image"]
    gt_mask = data["mask"]

    raw_slice, gt_mask = standardize_slice_and_mask(raw_slice, gt_mask)

    # Run Edge Inference
    is_offline = not sync_manager.is_online
    result: EdgePredictionResult = edge_engine.predict(
        raw_slice,
        patient_id=meta["patient_id"],
        slice_id=meta["sample_id"],
        is_offline=is_offline
    )

    # Log to local SQLite audit table
    sync_manager.log_inference(result)

    # Calculate Dice score against ground truth if available
    pred_m = result.segmentation_mask
    liver_intersection = np.sum((pred_m == 1) & (gt_mask == 1))
    liver_dice = float(2.0 * liver_intersection / (np.sum(pred_m == 1) + np.sum(gt_mask == 1) + 1e-7))

    tumor_gt_sum = np.sum(gt_mask == 2)
    tumor_pred_sum = np.sum(pred_m == 2)
    if tumor_gt_sum == 0 and tumor_pred_sum == 0:
        tumor_dice = 1.0
    else:
        tumor_intersection = np.sum((pred_m == 2) & (gt_mask == 2))
        tumor_dice = float(2.0 * tumor_intersection / (tumor_gt_sum + tumor_pred_sum + 1e-7))

    return {
        "prediction": result.to_dict(include_mask=False),
        "raw_slice_b64": array_to_base64_png(raw_slice),
        "colored_mask_b64": mask_to_rgba_base64_png(pred_m),
        "gt_mask_b64": mask_to_rgba_base64_png(gt_mask),
        "validation": {
            "liver_dice": round(liver_dice, 4),
            "tumor_dice": round(tumor_dice, 4),
            "mean_dice": round((liver_dice + tumor_dice) / 2.0, 4)
        },
        "case_title": meta["title"],
        "clinical_notes": meta["clinical_notes"]
    }


@app.post("/api/predict_upload")
async def predict_uploaded_file(
    file: UploadFile = File(...),
    patient_id: str = Form("UPLOAD-PATIENT")
):
    """Executes edge segmentation inference on a clinician-uploaded image."""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("L")
        pil_img = pil_img.resize((128, 128))
        arr = np.array(pil_img, dtype=np.float32) / 255.0

        is_offline = not sync_manager.is_online
        result = edge_engine.predict(
            arr,
            patient_id=patient_id,
            slice_id=file.filename or "UPLOAD_SLICE",
            is_offline=is_offline
        )

        sync_manager.log_inference(result)

        return {
            "prediction": result.to_dict(include_mask=False),
            "raw_slice_b64": array_to_base64_png(arr),
            "colored_mask_b64": mask_to_rgba_base64_png(result.segmentation_mask),
            "case_title": f"Custom Upload: {file.filename}",
            "clinical_notes": "Diagnostic analysis computed at edge node."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")


class ConnectivityToggleRequest(BaseModel):
    online: bool


@app.post("/api/toggle_offline")
def toggle_offline(req: ConnectivityToggleRequest):
    """Simulates disconnecting or restoring the edge clinic's broadband connection."""
    current = sync_manager.set_connectivity(req.online)
    return {
        "network_status": "ONLINE" if current else "OFFLINE",
        "message": "Edge clinic network connection updated.",
        "queue_status": sync_manager.get_queue_status()
    }


@app.post("/api/queue_simulation_update")
def queue_simulation_update():
    """Simulates a local client training step generating an FL model update while offline."""
    dummy_weights = {"weights": [np.random.randn(8, 8).astype(np.float32) for _ in range(2)]}
    update_id = sync_manager.queue_fl_update(
        client_id=1,
        round_id=sync_manager.get_queue_status()["pending_fl_updates_count"] + 1,
        num_samples=75,
        update_data=dummy_weights,
        payload_type="sparsified_topk"
    )
    return {
        "success": True,
        "update_id": update_id,
        "message": f"Local FL update #{update_id} buffered into SQLite WAL queue.",
        "queue_status": sync_manager.get_queue_status()
    }


@app.post("/api/sync_offline")
def sync_offline():
    """Triggers the FIFO replay and synchronization of all pending SQLite records."""
    summary = sync_manager.sync_all()
    return {
        "sync_summary": summary,
        "queue_status": sync_manager.get_queue_status()
    }


class ClinicianFeedbackRequest(BaseModel):
    patient_id: str
    slice_id: str
    clinician_id: str
    verified_diagnosis: str
    clinician_notes: str


@app.post("/api/feedback")
def submit_feedback(req: ClinicianFeedbackRequest):
    """Records clinician validation annotations and diagnostic approval."""
    fb_id = sync_manager.queue_clinical_feedback(
        patient_id=req.patient_id,
        slice_id=req.slice_id,
        clinician_id=req.clinician_id,
        verified_diagnosis=req.verified_diagnosis,
        clinician_notes=req.clinician_notes
    )
    return {
        "feedback_id": fb_id,
        "message": "Clinical feedback successfully recorded in audit log.",
        "queue_status": sync_manager.get_queue_status()
    }


@app.get("/api/fl_observatory")
def get_fl_observatory():
    """Supplies empirical benchmark data (Tables 4, 5, 7) and simulated hospital topologies."""
    # Table 4: Personalization Comparison
    table4_path = "results/table4_personalization_comparison.json"
    table4_data = []
    if os.path.exists(table4_path):
        with open(table4_path, "r") as f:
            table4_data = json.load(f)

    # Table 5: Privacy Sweep
    table5_path = "results/table5_privacy_sweep.json"
    table5_data = []
    if os.path.exists(table5_path):
        with open(table5_path, "r") as f:
            table5_data = json.load(f)

    # Table 7: Communication Efficiency
    table7_path = "results/table7_comm_efficiency.json"
    table7_data = []
    if os.path.exists(table7_path):
        with open(table7_path, "r") as f:
            table7_data = json.load(f)

    # Simulated Hospital Nodes Topology
    hospitals = [
        {
            "id": 0,
            "name": "Hospital A (St. Jude Regional)",
            "tier": "Tier-2 Regional Center",
            "samples": 120,
            "skew_profile": "High Tumor Skew (75% lesions)",
            "cluster_id": 1,
            "cfl_dice": 0.8124,
            "fedavg_dice": 0.4812,
            "status": "Active"
        },
        {
            "id": 1,
            "name": "Hospital B (Metro Liver Institute)",
            "tier": "Tertiary Referral Hub",
            "samples": 100,
            "skew_profile": "Mixed Pathologies (50% lesions)",
            "cluster_id": 1,
            "cfl_dice": 0.8045,
            "fedavg_dice": 0.4910,
            "status": "Active"
        },
        {
            "id": 2,
            "name": "Hospital C (Valley Health Clinic)",
            "tier": "Rural Edge Clinic",
            "samples": 80,
            "skew_profile": "Healthy Liver Skew (15% lesions)",
            "cluster_id": 2,
            "cfl_dice": 0.8048,
            "fedavg_dice": 0.4880,
            "status": "Active (Offline Sync Capable)"
        }
    ]

    return {
        "hospitals": hospitals,
        "table4_personalization": table4_data,
        "table5_privacy": table5_data,
        "table7_communication": table7_data
    }


@app.get("/api/audit_logs")
def get_audit_logs():
    """Returns recent diagnostic inference records from SQLite."""
    return {"inferences": sync_manager.get_recent_inferences(limit=25)}


# Mount static files for Frontend SPA (index.html, style.css, app.js)
FRONTEND_DIR = "frontend"
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("Starting FedLiverNet Telemedicine Platform on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
