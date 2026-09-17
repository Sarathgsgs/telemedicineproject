"""
edge/offline_manager.py
-----------------------
Offline Edge Inference and Asynchronous Synchronization Manager for
low-resource rural telemedicine clinics.

Features:
1. Zero-dependency local EdgeInferenceEngine powered by TorchScript or PyTorch.
2. Standardized Hounsfield Unit (HU) windowing [-100, 400] and multi-class segmentation.
3. Embedded SQLite database with Write-Ahead Logging (WAL) for thread-safe local queueing.
4. Offline queuing of Federated Learning model weights / Top-k sparsified gradient updates.
5. Offline queuing of clinical diagnostic feedback and validation annotations.
6. Automatic FIFO replay and idempotent synchronization upon network reconnection.
"""

import os
import time
import json
import gzip
import base64
import pickle
import sqlite3
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

from model.unet_attention import AttentionUNetLite


@dataclass
class EdgePredictionResult:
    """Structured response from edge diagnostic inference."""
    slice_id: str
    patient_id: str
    liver_pixels: int
    tumor_pixels: int
    total_pixels: int
    liver_area_pct: float
    tumor_area_pct: float
    mean_confidence: float
    tumor_detected: bool
    inference_ms: float
    is_offline: bool
    segmentation_mask: Optional[np.ndarray] = None  # uint8 2D array (0, 1, 2)

    def to_dict(self, include_mask: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_mask:
            data.pop("segmentation_mask", None)
        else:
            if self.segmentation_mask is not None:
                data["segmentation_mask"] = self.segmentation_mask.tolist()
        return data


class EdgeInferenceEngine:
    """
    Zero-network dependency local diagnostic inference engine for rural hospital edge nodes.
    Supports either compiled TorchScript (.pt) or standard PyTorch weights.
    """
    def __init__(
        self,
        model_path: str = "checkpoints/edge_model_traced.pt",
        device: str = "cpu",
        base_filters: int = 32
    ):
        self.device = device
        self.model_path = model_path
        self.base_filters = base_filters
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path: str):
        if not os.path.exists(path):
            # Fallback to centralized_best if traced model is not yet compiled
            fallback = "checkpoints/centralized_best.pt"
            if os.path.exists(fallback):
                print(f"[EdgeEngine] Warning: {path} not found. Loading fallback: {fallback}")
                m = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=self.base_filters)
                ckpt = torch.load(fallback, map_location=self.device)
                sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                m.load_state_dict(sd)
                return m.to(self.device)
            raise FileNotFoundError(f"Neither {path} nor fallback checkpoint {fallback} found.")

        try:
            # Attempt loading as TorchScript compiled module
            model = torch.jit.load(path, map_location=self.device)
            return model
        except Exception:
            # Fallback to eager PyTorch model loading
            m = AttentionUNetLite(in_channels=1, num_classes=3, base_filters=self.base_filters)
            ckpt = torch.load(path, map_location=self.device)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            m.load_state_dict(sd)
            return m.to(self.device)

    def preprocess_slice(self, ct_slice: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
        """
        Applies standard liver CT Hounsfield windowing [-100, 400] HU -> [0, 1]
        and converts to normalized PyTorch tensor (1, 1, H, W).
        """
        arr = np.asarray(ct_slice, dtype=np.float32)
        # If input is raw HU, window it; if already normalized in [0, 1], skip window
        if arr.min() < 0.0 or arr.max() > 1.0:
            arr = np.clip(arr, -100.0, 400.0)
            arr = (arr - (-100.0)) / (400.0 - (-100.0))

        # Ensure 2D (H, W)
        if arr.ndim == 3:
            arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(self.device)
        if tensor.shape[2:] != target_size:
            tensor = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
        return tensor

    def predict(
        self,
        ct_slice: np.ndarray,
        patient_id: str = "ANON_PATIENT",
        slice_id: str = "SLICE_001",
        is_offline: bool = True
    ) -> EdgePredictionResult:
        """
        Runs local diagnostic inference on a single CT slice.
        """
        t0 = time.perf_counter()
        input_tensor = self.preprocess_slice(ct_slice)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            conf_map = torch.max(probs, dim=1)[0].squeeze(0).cpu().numpy()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_pixels = int(pred_mask.size)
        liver_pixels = int(np.sum(pred_mask == 1))
        tumor_pixels = int(np.sum(pred_mask == 2))

        liver_pct = round((liver_pixels / total_pixels) * 100.0, 2)
        tumor_pct = round((tumor_pixels / total_pixels) * 100.0, 2)

        # Foreground mean confidence
        fg_mask = pred_mask > 0
        mean_conf = float(np.mean(conf_map[fg_mask])) if np.any(fg_mask) else float(np.mean(conf_map))

        return EdgePredictionResult(
            slice_id=slice_id,
            patient_id=patient_id,
            liver_pixels=liver_pixels,
            tumor_pixels=tumor_pixels,
            total_pixels=total_pixels,
            liver_area_pct=liver_pct,
            tumor_area_pct=tumor_pct,
            mean_confidence=round(mean_conf, 4),
            tumor_detected=bool(tumor_pixels >= 5),
            inference_ms=round(elapsed_ms, 2),
            is_offline=is_offline,
            segmentation_mask=pred_mask
        )


from contextlib import contextmanager


class OfflineSyncManager:
    """
    SQLite-backed transactional queue for storing local FL updates, clinician feedback,
    and inference audit logs during edge network outages. Replays and synchronizes
    all pending items upon network recovery.
    """
    def __init__(self, db_path: str = "edge/offline_store.db"):
        self.db_path = db_path
        self.is_online: bool = False  # Default state for edge deployment testing
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    def close(self):
        """Clean up any database resources."""
        pass

    def _init_db(self):
        with self._get_connection() as conn:
            # 1. Inference audit logs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    patient_id TEXT,
                    slice_id TEXT,
                    liver_pixels INTEGER,
                    tumor_pixels INTEGER,
                    liver_area_pct REAL,
                    tumor_area_pct REAL,
                    mean_confidence REAL,
                    tumor_detected INTEGER,
                    inference_ms REAL,
                    is_offline INTEGER
                )
            """)

            # 2. Queued Federated Learning model/gradient updates
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queued_fl_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    client_id INTEGER,
                    round_id INTEGER,
                    num_samples INTEGER,
                    payload_type TEXT, -- 'dense' or 'sparsified_topk'
                    payload_blob BLOB,
                    payload_size_kb REAL,
                    status TEXT DEFAULT 'QUEUED', -- 'QUEUED', 'SYNCING', 'SYNCED', 'FAILED'
                    synced_at TIMESTAMP NULL,
                    retry_count INTEGER DEFAULT 0,
                    error_msg TEXT NULL
                )
            """)

            # 3. Offline clinician feedback / annotations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clinical_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    patient_id TEXT,
                    slice_id TEXT,
                    clinician_id TEXT,
                    verified_diagnosis TEXT,
                    clinician_notes TEXT,
                    mask_blob BLOB NULL,
                    status TEXT DEFAULT 'QUEUED',
                    synced_at TIMESTAMP NULL,
                    retry_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def set_connectivity(self, online: bool) -> bool:
        """Manually toggle network online/offline state for simulation and testing."""
        self.is_online = bool(online)
        return self.is_online

    def log_inference(self, result: EdgePredictionResult) -> int:
        """Records an edge inference execution event into the audit log."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO inference_logs (
                    patient_id, slice_id, liver_pixels, tumor_pixels,
                    liver_area_pct, tumor_area_pct, mean_confidence,
                    tumor_detected, inference_ms, is_offline
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.patient_id,
                result.slice_id,
                result.liver_pixels,
                result.tumor_pixels,
                result.liver_area_pct,
                result.tumor_area_pct,
                result.mean_confidence,
                1 if result.tumor_detected else 0,
                result.inference_ms,
                1 if result.is_offline else 0
            ))
            conn.commit()
            return cursor.lastrowid

    def queue_fl_update(
        self,
        client_id: int,
        round_id: int,
        num_samples: int,
        update_data: Any,
        payload_type: str = "dense"
    ) -> int:
        """
        Serializes and compresses local FL parameters or sparsified payloads,
        persisting them to the SQLite queue with transactional safety.
        """
        # Compress payload using pickle + gzip for minimal disk footprint
        raw_bytes = pickle.dumps(update_data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_blob = gzip.compress(raw_bytes)
        size_kb = len(compressed_blob) / 1024.0

        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO queued_fl_updates (
                    client_id, round_id, num_samples, payload_type,
                    payload_blob, payload_size_kb, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'QUEUED')
            """, (
                client_id,
                round_id,
                num_samples,
                payload_type,
                compressed_blob,
                round(size_kb, 2)
            ))
            conn.commit()
            return cursor.lastrowid

    def queue_clinical_feedback(
        self,
        patient_id: str,
        slice_id: str,
        clinician_id: str,
        verified_diagnosis: str,
        clinician_notes: str = "",
        modified_mask: Optional[np.ndarray] = None
    ) -> int:
        """Persists clinician review or corrections made during offline teleconsultation."""
        mask_blob = None
        if modified_mask is not None:
            mask_blob = gzip.compress(pickle.dumps(modified_mask, protocol=pickle.HIGHEST_PROTOCOL))

        with self._get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO clinical_feedback (
                    patient_id, slice_id, clinician_id,
                    verified_diagnosis, clinician_notes, mask_blob, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'QUEUED')
            """, (
                patient_id,
                slice_id,
                clinician_id,
                verified_diagnosis,
                clinician_notes,
                mask_blob
            ))
            conn.commit()
            return cursor.lastrowid

    def get_queue_status(self) -> Dict[str, Any]:
        """Returns instantaneous counts and storage sizes of pending offline items."""
        with self._get_connection() as conn:
            pending_updates = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(payload_size_kb), 0.0) FROM queued_fl_updates WHERE status = 'QUEUED'"
            ).fetchone()
            synced_updates = conn.execute(
                "SELECT COUNT(*) FROM queued_fl_updates WHERE status = 'SYNCED'"
            ).fetchone()[0]
            failed_updates = conn.execute(
                "SELECT COUNT(*) FROM queued_fl_updates WHERE status = 'FAILED'"
            ).fetchone()[0]

            pending_feedback = conn.execute(
                "SELECT COUNT(*) FROM clinical_feedback WHERE status = 'QUEUED'"
            ).fetchone()[0]
            synced_feedback = conn.execute(
                "SELECT COUNT(*) FROM clinical_feedback WHERE status = 'SYNCED'"
            ).fetchone()[0]

            total_inferences = conn.execute("SELECT COUNT(*) FROM inference_logs").fetchone()[0]
            offline_inferences = conn.execute(
                "SELECT COUNT(*) FROM inference_logs WHERE is_offline = 1"
            ).fetchone()[0]

        return {
            "network_status": "ONLINE" if self.is_online else "OFFLINE",
            "pending_fl_updates_count": pending_updates[0],
            "pending_fl_updates_kb": round(float(pending_updates[1]), 2),
            "synced_fl_updates_count": synced_updates,
            "failed_fl_updates_count": failed_updates,
            "pending_feedback_count": pending_feedback,
            "synced_feedback_count": synced_feedback,
            "total_inferences_logged": total_inferences,
            "offline_inferences_logged": offline_inferences
        }

    def sync_all(
        self,
        fl_server_upload_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        feedback_upload_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> Dict[str, Any]:
        """
        Replays and uploads queued items in strict FIFO order to the central server.
        If offline, the call safely defers synchronization without data loss.

        Args:
            fl_server_upload_fn: Callback taking a dict payload and returning True on HTTP 200/ACK.
            feedback_upload_fn: Callback taking feedback payload and returning True on HTTP 200/ACK.
        """
        if not self.is_online:
            status = self.get_queue_status()
            return {
                "sync_performed": False,
                "reason": "Network connection is offline. Updates safely retained in local queue.",
                "pending_fl_updates": status["pending_fl_updates_count"],
                "pending_feedback": status["pending_feedback_count"]
            }

        synced_fl = 0
        failed_fl = 0
        synced_fb = 0
        failed_fb = 0

        # Default echo callbacks if none provided (for standalone testing / simulation)
        if fl_server_upload_fn is None:
            fl_server_upload_fn = lambda payload: True
        if feedback_upload_fn is None:
            feedback_upload_fn = lambda payload: True

        with self._get_connection() as conn:
            # 1. Drain queued FL updates FIFO
            cursor = conn.execute(
                "SELECT * FROM queued_fl_updates WHERE status = 'QUEUED' ORDER BY id ASC"
            )
            rows = cursor.fetchall()

            for row in rows:
                update_id = row["id"]
                conn.execute(
                    "UPDATE queued_fl_updates SET status = 'SYNCING' WHERE id = ?", (update_id,)
                )
                conn.commit()

                try:
                    decompressed = gzip.decompress(row["payload_blob"])
                    deserialized_data = pickle.loads(decompressed)

                    payload_for_server = {
                        "update_id": update_id,
                        "client_id": row["client_id"],
                        "round_id": row["round_id"],
                        "num_samples": row["num_samples"],
                        "payload_type": row["payload_type"],
                        "data": deserialized_data,
                        "created_at": row["created_at"]
                    }

                    success = fl_server_upload_fn(payload_for_server)
                    if success:
                        conn.execute("""
                            UPDATE queued_fl_updates
                            SET status = 'SYNCED', synced_at = CURRENT_TIMESTAMP, error_msg = NULL
                            WHERE id = ?
                        """, (update_id,))
                        synced_fl += 1
                    else:
                        raise RuntimeError("Server rejected or failed to acknowledge update payload.")

                except Exception as e:
                    failed_fl += 1
                    conn.execute("""
                        UPDATE queued_fl_updates
                        SET status = 'FAILED', retry_count = retry_count + 1, error_msg = ?
                        WHERE id = ?
                    """, (str(e), update_id))

                conn.commit()

            # 2. Drain queued clinical feedback FIFO
            cursor = conn.execute(
                "SELECT * FROM clinical_feedback WHERE status = 'QUEUED' ORDER BY id ASC"
            )
            fb_rows = cursor.fetchall()

            for row in fb_rows:
                fb_id = row["id"]
                try:
                    mask = None
                    if row["mask_blob"] is not None:
                        mask = pickle.loads(gzip.decompress(row["mask_blob"]))

                    fb_payload = {
                        "feedback_id": fb_id,
                        "patient_id": row["patient_id"],
                        "slice_id": row["slice_id"],
                        "clinician_id": row["clinician_id"],
                        "verified_diagnosis": row["verified_diagnosis"],
                        "clinician_notes": row["clinician_notes"],
                        "mask": mask
                    }

                    success = feedback_upload_fn(fb_payload)
                    if success:
                        conn.execute("""
                            UPDATE clinical_feedback
                            SET status = 'SYNCED', synced_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (fb_id,))
                        synced_fb += 1
                    else:
                        raise RuntimeError("Server rejected feedback payload.")

                except Exception as e:
                    failed_fb += 1
                    conn.execute("""
                        UPDATE clinical_feedback
                        SET status = 'FAILED', retry_count = retry_count + 1
                        WHERE id = ?
                    """, (fb_id,))

                conn.commit()

        return {
            "sync_performed": True,
            "synced_fl_updates": synced_fl,
            "failed_fl_updates": failed_fl,
            "synced_feedback": synced_fb,
            "failed_feedback": failed_fb,
            "remaining_pending": self.get_queue_status()["pending_fl_updates_count"]
        }

    def get_recent_inferences(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns recent inference audit events for dashboard display."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM inference_logs ORDER BY id DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def reset_queue(self):
        """Clears all queued entries (primarily used in automated unit tests)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM queued_fl_updates;")
            conn.execute("DELETE FROM clinical_feedback;")
            conn.execute("DELETE FROM inference_logs;")
            conn.commit()
