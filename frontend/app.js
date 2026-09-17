/**
 * FedLiverNet Telemedicine Platform - Master Client Application
 * Handles:
 * - Dual-canvas CT slice and multi-class segmentation mask rendering.
 * - Dynamic opacity adjustments, channel toggling, and pixel inspection.
 * - Quick-pick sample loading and custom file drag-and-drop inference.
 * - Real-time metrics updating and diagnostic verdict alarms.
 * - Edge offline simulation and SQLite sync replay management.
 * - Federated Learning Observatory (CFL-Lite, Differential Privacy, Top-k Sparsification).
 */

const API_BASE = "";

// Application State
const state = {
  activeView: "teleconsultView",
  currentSampleId: "slice_0001",
  isOffline: false,
  showLiver: true,
  showTumor: true,
  showGt: false,
  maskOpacity: 0.8,
  rawImage: null,
  predMaskImage: null,
  gtMaskImage: null,
  currentPrediction: null,
  dpSettings: {
    1: { eps: 0.1, sigma: 48.4481, label: "Extreme Privacy (\u03b5 = 0.1)", desc: "High theoretical privacy; aggressive Gaussian noise perturbation." },
    2: { eps: 0.5, sigma: 9.6896, label: "Strong Privacy (\u03b5 = 0.5)", desc: "Strong privacy guarantee; minor boundary attenuation." },
    3: { eps: 1.0, sigma: 4.8448, label: "Balanced Standard (\u03b5 = 1.0)", desc: "Clinical deployment sweet spot: prevents deep reconstruction attacks while preserving full diagnostic fidelity." },
    4: { eps: 5.0, sigma: 0.9690, label: "Relaxed Privacy (\u03b5 = 5.0)", desc: "High segmentation fidelity with minimal gradient noise." },
    5: { eps: Infinity, sigma: 0.0000, label: "No DP (\u03b5 = \u221e)", desc: "Unperturbed raw gradients; zero added Gaussian noise." }
  }
};

// DOM Elements
const elements = {
  // Navigation
  tabTeleconsult: document.getElementById("tabTeleconsult"),
  tabObservatory: document.getElementById("tabObservatory"),
  tabAudit: document.getElementById("tabAudit"),
  teleconsultView: document.getElementById("teleconsultView"),
  observatoryView: document.getElementById("observatoryView"),
  auditView: document.getElementById("auditView"),
  connectivityBadge: document.getElementById("connectivityBadge"),
  connectivityText: document.getElementById("connectivityText"),
  btnToggleOffline: document.getElementById("btnToggleOffline"),
  btnToggleOfflineText: document.getElementById("btnToggleOfflineText"),

  // Case Selector & Upload
  caseCards: document.querySelectorAll(".case-card"),
  activeCaseBadge: document.getElementById("activeCaseBadge"),
  uploadDropzone: document.getElementById("uploadDropzone"),
  fileInput: document.getElementById("fileInput"),

  // Canvas
  ctCanvas: document.getElementById("ctCanvas"),
  maskCanvas: document.getElementById("maskCanvas"),
  maskOpacitySlider: document.getElementById("maskOpacitySlider"),
  maskOpacityVal: document.getElementById("maskOpacityVal"),
  btnToggleLiver: document.getElementById("btnToggleLiver"),
  btnToggleTumor: document.getElementById("btnToggleTumor"),
  btnToggleGt: document.getElementById("btnToggleGt"),
  btnRunInference: document.getElementById("btnRunInference"),
  lblOverlayMode: document.getElementById("lblOverlayMode"),
  viewportCoords: document.getElementById("viewportCoords"),

  // Metrics & Verdict
  verdictBanner: document.getElementById("verdictBanner"),
  verdictTitle: document.getElementById("verdictTitle"),
  verdictSubtitle: document.getElementById("verdictSubtitle"),
  valTumorPct: document.getElementById("valTumorPct"),
  tumorPixelBadge: document.getElementById("tumorPixelBadge"),
  valLiverPct: document.getElementById("valLiverPct"),
  liverPixelBadge: document.getElementById("liverPixelBadge"),
  valConfidence: document.getElementById("valConfidence"),
  valLatency: document.getElementById("valLatency"),
  valTumorDice: document.getElementById("valTumorDice"),
  valLiverDice: document.getElementById("valLiverDice"),
  valMeanDice: document.getElementById("valMeanDice"),

  // Offline Sync
  statPendingUpdates: document.getElementById("statPendingUpdates"),
  statPendingFeedback: document.getElementById("statPendingFeedback"),
  statOfflineInferences: document.getElementById("statOfflineInferences"),
  lblSyncMode: document.getElementById("lblSyncMode"),
  btnSimulateFLBatch: document.getElementById("btnSimulateFLBatch"),
  btnSyncNow: document.getElementById("btnSyncNow"),

  // Clinician Notes
  txtClinicianNotes: document.getElementById("txtClinicianNotes"),
  btnApproveDiagnosis: document.getElementById("btnApproveDiagnosis"),
  btnSubmitFeedback: document.getElementById("btnSubmitFeedback"),

  // Observatory & Audit
  hospitalNodesList: document.getElementById("hospitalNodesList"),
  sliderEpsilon: document.getElementById("sliderEpsilon"),
  badgeEpsilonVal: document.getElementById("badgeEpsilonVal"),
  dpFormulaDisplay: document.getElementById("dpFormulaDisplay"),
  dpExplanationText: document.getElementById("dpExplanationText"),
  auditTableBody: document.getElementById("auditTableBody"),
  btnRefreshAudit: document.getElementById("btnRefreshAudit"),
  toastContainer: document.getElementById("toastContainer")
};

// Canvas 2D Rendering Contexts
const ctCtx = elements.ctCanvas.getContext("2d");
const maskCtx = elements.maskCanvas.getContext("2d");

// -------------------------------------------------------------
// Toast Notifications Helper
// -------------------------------------------------------------
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = "toast";
  let icon = `\u2713`;
  if (type === "warning") icon = `\u26a0`;
  if (type === "error") icon = `\u2716`;

  toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
  elements.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// -------------------------------------------------------------
// View / Tab Switching
// -------------------------------------------------------------
function setupNavigation() {
  const navBtns = [elements.tabTeleconsult, elements.tabObservatory, elements.tabAudit];
  const views = [elements.teleconsultView, elements.observatoryView, elements.auditView];

  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      navBtns.forEach(b => b.classList.remove("active"));
      views.forEach(v => v.classList.remove("active"));

      btn.classList.add("active");
      const targetViewId = btn.getAttribute("data-view");
      const targetView = document.getElementById(targetViewId);
      if (targetView) targetView.classList.add("active");

      if (targetViewId === "observatoryView") loadObservatoryData();
      if (targetViewId === "auditView") loadAuditLogs();
    });
  });
}

// -------------------------------------------------------------
// Canvas Drawing & Overlay Rendering
// -------------------------------------------------------------
function renderCanvases() {
  // Clear canvases
  ctCtx.clearRect(0, 0, 128, 128);
  maskCtx.clearRect(0, 0, 128, 128);

  // 1. Draw raw CT slice
  if (state.rawImage) {
    ctCtx.drawImage(state.rawImage, 0, 0, 128, 128);
  }

  // 2. Draw mask overlay (either predicted or ground-truth)
  const maskImg = state.showGt ? state.gtMaskImage : state.predMaskImage;
  if (maskImg) {
    maskCtx.globalAlpha = state.maskOpacity;
    maskCtx.drawImage(maskImg, 0, 0, 128, 128);
  }
}

function updateCrosshair(e) {
  const rect = elements.ctCanvas.getBoundingClientRect();
  const scaleX = 128 / rect.width;
  const scaleY = 128 / rect.height;
  const x = Math.floor((e.clientX - rect.left) * scaleX);
  const y = Math.floor((e.clientY - rect.top) * scaleY);

  if (x >= 0 && x < 128 && y >= 0 && y < 128) {
    const pixel = ctCtx.getImageData(x, y, 1, 1).data;
    const intensity = (pixel[0] / 255.0).toFixed(2);
    elements.viewportCoords.textContent = `X: ${x} | Y: ${y} | Intensity: ${intensity}`;
  }
}

// -------------------------------------------------------------
// Load and Predict Clinical Sample
// -------------------------------------------------------------
async function loadSamplePrediction(sampleId) {
  try {
    elements.activeCaseBadge.textContent = "Computing Edge Inference...";
    const res = await fetch(`${API_BASE}/api/predict_sample`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: sampleId })
    });

    if (!res.ok) throw new Error("Inference request failed");
    const data = await res.json();
    state.currentPrediction = data;

    // Load CT Image
    const rawImg = new Image();
    rawImg.onload = () => {
      state.rawImage = rawImg;
      renderCanvases();
    };
    rawImg.src = data.raw_slice_b64;

    // Load Predicted Mask Image
    const predImg = new Image();
    predImg.onload = () => {
      state.predMaskImage = predImg;
      renderCanvases();
    };
    predImg.src = data.colored_mask_b64;

    // Load Ground Truth Mask Image
    const gtImg = new Image();
    gtImg.onload = () => {
      state.gtMaskImage = gtImg;
      renderCanvases();
    };
    gtImg.src = data.gt_mask_b64;

    // Update UI Metrics
    updateMetricsDisplay(data);
    refreshQueueStatus();

  } catch (err) {
    console.error(err);
    showToast("Error loading clinical sample: " + err.message, "error");
  }
}

// -------------------------------------------------------------
// Update Quantitative Metrics & Verdict
// -------------------------------------------------------------
function updateMetricsDisplay(data) {
  const p = data.prediction;
  const v = data.validation;

  elements.activeCaseBadge.textContent = data.case_title || p.slice_id;
  elements.valTumorPct.textContent = `${p.tumor_area_pct} %`;
  elements.tumorPixelBadge.textContent = `${p.tumor_pixels.toLocaleString()} px`;

  elements.valLiverPct.textContent = `${p.liver_area_pct} %`;
  elements.liverPixelBadge.textContent = `${p.liver_pixels.toLocaleString()} px`;

  elements.valConfidence.textContent = `${(p.mean_confidence * 100).toFixed(1)} %`;
  elements.valLatency.textContent = `${p.inference_ms} ms`;

  // Verdict Banner
  if (p.tumor_detected) {
    elements.verdictBanner.className = "verdict-banner critical";
    elements.verdictTitle.textContent = "CRITICAL: Focal HCC Lesion Detected";
    elements.verdictSubtitle.textContent = `Lesion surface area: ${p.tumor_area_pct}% (${p.tumor_pixels} pixels). Urgent hepatic teleconsultation recommended.`;
  } else {
    elements.verdictBanner.className = "verdict-banner normal";
    elements.verdictTitle.textContent = "NORMAL: Healthy Hepatic Parenchyma";
    elements.verdictSubtitle.textContent = `Parenchymal volume fraction: ${p.liver_area_pct}%. No discrete focal lesion or hypervascular neoplasm detected.`;
  }

  // Dice Validation
  if (v) {
    elements.valTumorDice.textContent = v.tumor_dice.toFixed(4);
    elements.valLiverDice.textContent = v.liver_dice.toFixed(4);
    elements.valMeanDice.textContent = v.mean_dice.toFixed(4);
  }

  // Set suggested clinical note
  if (data.clinical_notes) {
    elements.txtClinicianNotes.value = `[Edge AI Assistant Staging] ${data.clinical_notes} Tumor Area: ${p.tumor_area_pct}%, Conf: ${(p.mean_confidence*100).toFixed(1)}%.`;
  }
}

// -------------------------------------------------------------
// Offline Mode Simulation & Synchronization
// -------------------------------------------------------------
async function toggleOfflineState() {
  try {
    const nextState = !state.isOffline;
    const res = await fetch(`${API_BASE}/api/toggle_offline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ online: nextState })
    });

    const data = await res.json();
    state.isOffline = !nextState;

    if (state.isOffline) {
      elements.connectivityBadge.className = "status-badge offline";
      elements.connectivityText.textContent = "EDGE OFFLINE";
      elements.btnToggleOfflineText.textContent = "Restore Connection";
      elements.btnToggleOffline.className = "btn-toggle-offline online-mode";
      showToast("Broadband outage simulated: Edge station operating offline with local SQLite WAL buffering.", "warning");
    } else {
      elements.connectivityBadge.className = "status-badge";
      elements.connectivityText.textContent = "EDGE ONLINE";
      elements.btnToggleOfflineText.textContent = "Simulate Outage";
      elements.btnToggleOffline.className = "btn-toggle-offline";
      showToast("Broadband restored: Central FL aggregation coordinator reachable.", "info");
    }

    refreshQueueStatus();
  } catch (err) {
    console.error(err);
    showToast("Failed to toggle connectivity: " + err.message, "error");
  }
}

async function simulateLocalFLBatch() {
  try {
    const res = await fetch(`${API_BASE}/api/queue_simulation_update`, { method: "POST" });
    const data = await res.json();
    showToast(`Local FL round update #${data.update_id} buffered to SQLite queue.`, "info");
    refreshQueueStatus();
  } catch (err) {
    showToast("Error queueing FL update: " + err.message, "error");
  }
}

async function syncPendingRecords() {
  try {
    elements.lblSyncMode.textContent = "SYNCING...";
    elements.btnSyncNow.disabled = true;

    const res = await fetch(`${API_BASE}/api/sync_offline`, { method: "POST" });
    const data = await res.json();
    const s = data.sync_summary;

    if (s.sync_performed) {
      showToast(`Sync complete! Flushed ${s.synced_fl_updates} model updates & ${s.synced_feedback} clinical notes to central coordinator.`, "info");
    } else {
      showToast(s.reason || "Sync deferred (Offline mode).", "warning");
    }

    elements.lblSyncMode.textContent = "IDLE";
    elements.btnSyncNow.disabled = false;
    refreshQueueStatus();
  } catch (err) {
    elements.lblSyncMode.textContent = "ERROR";
    elements.btnSyncNow.disabled = false;
    showToast("Sync failed: " + err.message, "error");
  }
}

async function refreshQueueStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    const data = await res.json();
    const q = data.offline_queue;

    elements.statPendingUpdates.textContent = q.pending_fl_updates_count;
    elements.statPendingFeedback.textContent = q.pending_feedback_count;
    elements.statOfflineInferences.textContent = q.offline_inferences_logged;

    state.isOffline = (data.edge_connectivity === "OFFLINE");
  } catch (err) {
    console.warn("Status refresh error:", err);
  }
}

// -------------------------------------------------------------
// Custom File Upload
// -------------------------------------------------------------
function setupUploadHandlers() {
  elements.uploadDropzone.addEventListener("click", () => elements.fileInput.click());

  elements.fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("patient_id", "UPLOAD-PATIENT-CUSTOM");

    try {
      showToast("Uploading and computing edge segmentation...", "info");
      const res = await fetch(`${API_BASE}/api/predict_upload`, {
        method: "POST",
        body: formData
      });

      if (!res.ok) throw new Error("Upload inference failed");
      const data = await res.json();
      state.currentPrediction = data;

      const rawImg = new Image();
      rawImg.onload = () => { state.rawImage = rawImg; renderCanvases(); };
      rawImg.src = data.raw_slice_b64;

      const predImg = new Image();
      predImg.onload = () => { state.predMaskImage = predImg; renderCanvases(); };
      predImg.src = data.colored_mask_b64;

      updateMetricsDisplay(data);
      showToast("Custom CT slice analyzed successfully!", "info");
      refreshQueueStatus();
    } catch (err) {
      showToast("Upload error: " + err.message, "error");
    }
  });
}

// -------------------------------------------------------------
// FL Observatory Data Loading
// -------------------------------------------------------------
async function loadObservatoryData() {
  try {
    const res = await fetch(`${API_BASE}/api/fl_observatory`);
    const data = await res.json();

    // Render Hospital Nodes
    elements.hospitalNodesList.innerHTML = "";
    data.hospitals.forEach(h => {
      const card = document.createElement("div");
      card.className = "hospital-node-card";
      card.innerHTML = `
        <div class="hospital-info">
          <h4>${h.name}</h4>
          <p>${h.tier} &bull; ${h.samples} Local LiTS Slices &bull; ${h.skew_profile}</p>
        </div>
        <div>
          <span class="cluster-pill ${h.cluster_id === 1 ? 'cluster-1' : 'cluster-2'}">
            Cluster ${h.cluster_id} (CFL Dice: ${h.cfl_dice})
          </span>
        </div>
      `;
      elements.hospitalNodesList.appendChild(card);
    });

  } catch (err) {
    console.error("Observatory data error:", err);
  }
}

// Dynamic Epsilon Slider
function setupObservatoryControls() {
  elements.sliderEpsilon.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    const setting = state.dpSettings[val];

    elements.badgeEpsilonVal.textContent = setting.label;
    elements.dpFormulaDisplay.textContent = `DP-SGD: C = 1.0 \u2022 Gaussian Noise \u03c3 = ${setting.sigma.toFixed(4)} \u2022 Analytical \u03b4 = 1e-5`;
    elements.dpExplanationText.textContent = setting.desc;
  });
}

// -------------------------------------------------------------
// Audit Log Viewer
// -------------------------------------------------------------
async function loadAuditLogs() {
  try {
    const res = await fetch(`${API_BASE}/api/audit_logs`);
    const data = await res.json();

    elements.auditTableBody.innerHTML = "";
    if (!data.inferences || data.inferences.length === 0) {
      elements.auditTableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No inference records logged yet.</td></tr>`;
      return;
    }

    data.inferences.forEach(item => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>#${item.id}</td>
        <td>${item.created_at || 'Just now'}</td>
        <td><strong>${item.patient_id}</strong></td>
        <td>${item.slice_id}</td>
        <td>${item.tumor_area_pct}%</td>
        <td>${(item.mean_confidence * 100).toFixed(1)}%</td>
        <td>${item.inference_ms} ms</td>
        <td>
          <span class="overlay-badge" style="${item.is_offline ? 'background: rgba(239,68,68,0.2); color:#FCA5A5;' : 'background: rgba(16,185,129,0.2); color:#A7F3D0;'}">
            ${item.is_offline ? 'OFFLINE' : 'ONLINE'}
          </span>
        </td>
      `;
      elements.auditTableBody.appendChild(tr);
    });

  } catch (err) {
    console.error("Audit log error:", err);
  }
}

// -------------------------------------------------------------
// Initial Setup & Event Listeners
// -------------------------------------------------------------
function initializeApp() {
  setupNavigation();
  setupUploadHandlers();
  setupObservatoryControls();

  // Opacity Slider
  elements.maskOpacitySlider.addEventListener("input", (e) => {
    state.maskOpacity = parseInt(e.target.value) / 100.0;
    elements.maskOpacityVal.textContent = `${e.target.value}%`;
    renderCanvases();
  });

  // Crosshair coordinate tracking
  elements.ctCanvas.parentElement.addEventListener("mousemove", updateCrosshair);

  // Quick Case Cards Click
  elements.caseCards.forEach(card => {
    card.addEventListener("click", () => {
      elements.caseCards.forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      const sampleId = card.getAttribute("data-sample");
      state.currentSampleId = sampleId;
      loadSamplePrediction(sampleId);
    });
  });

  // Ground Truth Toggle
  elements.btnToggleGt.addEventListener("click", () => {
    state.showGt = !state.showGt;
    elements.btnToggleGt.classList.toggle("active", state.showGt);
    elements.lblOverlayMode.textContent = state.showGt ? "GROUND TRUTH OVERLAY" : "PREDICTED OVERLAY";
    renderCanvases();
  });

  // Re-run inference button
  elements.btnRunInference.addEventListener("click", () => {
    loadSamplePrediction(state.currentSampleId);
  });

  // Offline Buttons
  elements.btnToggleOffline.addEventListener("click", toggleOfflineState);
  elements.btnSimulateFLBatch.addEventListener("click", simulateLocalFLBatch);
  elements.btnSyncNow.addEventListener("click", syncPendingRecords);
  elements.btnRefreshAudit.addEventListener("click", loadAuditLogs);

  // Clinician notes buttons
  elements.btnApproveDiagnosis.addEventListener("click", () => {
    elements.txtClinicianNotes.value += " [VERIFIED BY RADIOLOGIST: Segmentation approved for teleconsultation record.]";
    showToast("Diagnosis verified and stamped.", "info");
  });

  elements.btnSubmitFeedback.addEventListener("click", async () => {
    try {
      const res = await fetch(`${API_BASE}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: state.currentPrediction?.prediction?.patient_id || "PATIENT",
          slice_id: state.currentSampleId,
          clinician_id: "DR_CONSULTANT_01",
          verified_diagnosis: state.currentPrediction?.prediction?.tumor_detected ? "HCC Positive" : "Normal Parenchyma",
          clinician_notes: elements.txtClinicianNotes.value
        })
      });
      showToast("Clinical teleconsultation notes logged to SQLite audit database.", "info");
      refreshQueueStatus();
    } catch (err) {
      showToast("Error saving notes: " + err.message, "error");
    }
  });

  // Initial Load
  loadSamplePrediction(state.currentSampleId);
  refreshQueueStatus();
}

// Boot application when DOM is ready
document.addEventListener("DOMContentLoaded", initializeApp);
