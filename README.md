# FedLiverNet: Privacy-Preserving Federated Learning Telemedicine Platform for AI-Assisted Liver Disease Screening

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower FL 1.7+](https://img.shields.io/badge/Flower%20FL-1.7%2B-ffd43b.svg)](https://flower.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-57%20Passed%20(100%25)-success.svg)]()

> **Final Year Computer Science and Engineering Project (Subject Code: 23CS8701)**  
> **Presenter:** YUVARAJAN D (Reg. No: 421123102175), B.E. CSE  
> **Supervisor:** Mrs. MANIMEGALAI, Assistant Professor, Department of Computer Science and Engineering  

---

## 📌 Executive Overview

**FedLiverNet** is a privacy-preserving, edge-resilient federated telemedicine platform engineered for AI-assisted hepatic lesion and hepatocellular carcinoma (HCC) screening in rural and semi-urban primary healthcare centers.

Under real-world multi-institutional deployments, clinical data is non-IID (pathology skew between tertiary cancer hubs and rural screening outposts). Vanilla federated averaging (`FedAvg`) collapses on minority tumor tissue due to conflicting gradients. **FedLiverNet** resolves this via **Clustered Federated Learning (CFL-Lite)**, **Client-Level Differential Privacy**, **Zero-Sum Additive Masking**, **Top-20% Gradient Sparsification**, and **SQLite WAL Store-and-Forward Offline Resilience**.

---

## 🔬 Empirical Benchmark Results (LiTS 2017 Real CT Cohort)

Evaluated under strict patient-level isolation (`subject_id` split with 0% slice leakage):

| Learning Paradigm | Mean Dice | Liver Dice | Tumor Dice | Tumor HD95 (mm) | Inter-Client Variance ($\sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline (Upper Bound)** | **0.7410** | **0.6447** | **0.8372** | **16.28 mm** | 0.0000 |
| **Standard FedAvg (Non-IID Collaborative)** | 0.4173 | 0.4102 | 0.4244 | 57.56 mm | 0.0000 |
| **FedAvg + CFL-Lite (Personalized Flagship)** | **0.6405** | **0.5135** | **0.7674** | **22.62 mm** | **0.1023** |

* **Lesion Recovery:** Tumor Dice surges from **0.4244 to 0.7674** (+34.3% recovery), eliminating catastrophic lesion forgetting.
* **Boundary Precision:** 95th Percentile Hausdorff Distance decreases from **57.56 mm down to 22.62 mm** (-60.7% boundary error reduction).
* **Communication Savings:** Top-20% gradient sparsification saves **79.6% transmission bandwidth** (13.84 MB $\to$ 2.82 MB per round).
* **Differential Privacy:** Analytical Rényi DP composition reports a certified privacy budget of $(\epsilon=6.42, \delta=10^{-5})$ over 3 rounds.
* **Zero Raw Data Transfer:** Exactly **0 bytes of raw CT images** leave any hospital client.

---

## 🚀 Quickstart Guide (Running on Any Machine)

### 1. Prerequisites
* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
* **Python:** 3.10, 3.11, or 3.12 (64-bit)
* **Hardware:** Any commodity laptop or desktop (Intel Core i5/i7 or AMD Ryzen, 8 GB+ RAM). No GPU required; edge model runs on CPU in $<120$ ms.

### 2. Clone the Repository
```bash
git clone https://github.com/Sarathgsgs/telemedicineproject.git
cd telemedicineproject
```

### 3. Create & Activate Virtual Environment
* **Windows (Command Prompt / PowerShell):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
* **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Platform (One-Click)
* **Windows:**
  Double-click `start.bat` or run:
  ```cmd
  .\start.bat
  ```
* **Linux / macOS:**
  ```bash
  python run.py
  ```

The launcher will verify checkpoints, boot the FastAPI backend, and automatically open your default browser to:
👉 **`http://127.0.0.1:8000`**

---

## 🧪 Testing and Verification

### Run the Full Automated Test Suite (57 Tests)
```bash
pytest tests/ -v
```
All 57 tests across unit, FL, privacy, edge export, and offline resilience modules run in $<120$ seconds.

### Run the Interactive Clinical Demonstration
```bash
python demo/run_clinical_demo.py
```
Steps through multi-hospital non-IID setup, TorchScript CPU inference, DP-masked gradient exchange, and offline SQLite WAL store-and-forward queueing.

### Run the Fair Comparative Benchmark
```bash
python experiments/run_fair_comparison.py
```
Outputs reproducible metrics to `results/real_lits_fair_comparison.json`.

---

## 📂 Project Architecture

```
telemedicine platform/
├── backend/
│   ├── main.py                     # FastAPI REST API & static webserver
├── checkpoints/
│   └── edge_model_traced.pt        # Pre-compiled TorchScript CPU model (31 MB)
├── comm/
│   ├── compression.py              # Top-20% gradient sparsification & error feedback
│   └── topk_sparsify.py            # Sparsification operator
├── data/
│   ├── dataset.py                  # PyTorch Dataset for 2D axial CT slices
│   ├── init_dataset.py             # Slicing & soft-tissue windowing [-100, 400] HU
│   ├── lits_downloader.py          # LiTS NIfTI parser & volume matcher
│   ├── patient_splitter.py         # Leakage-free patient-level partitioner
│   └── synthetic_generator.py      # Synthetic backup cohort generator
├── demo/
│   └── run_clinical_demo.py        # 4-stage interactive clinical walkthrough
├── docs/
│   ├── THREAT_MODEL.md             # Formal threat model & HIPAA/GDPR analysis
│   └── DISEASE_TAXONOMY.md         # Clinical classification of liver lesions
├── edge/
│   ├── model_exporter.py           # TorchScript tracing & latency benchmark
│   ├── offline_manager.py          # Unified edge engine & offline sync
│   └── offline_sync.py             # SQLite WAL store-and-forward queue
├── experiments/
│   ├── run_fair_comparison.py      # Real LiTS comparative benchmark script
│   └── run_privacy_sweep.py        # Privacy budget sweep (epsilon vs Dice)
├── fl/
│   ├── cfl_clustering.py           # Cosine similarity matrix & clustering
│   ├── client.py                   # Flower hospital client + local adaptation
│   └── server.py                   # Custom Flower aggregation strategy
├── frontend/
│   ├── index.html                  # Clinical teleconsultation web station
│   ├── style.css                   # Responsive clinical theme
│   └── app.js                      # Client logic & FL observatory visualization
├── model/
│   ├── losses.py                   # CompositeLoss (Dice + Weighted CE + Focal)
│   ├── metrics.py                  # Multi-class Dice, IoU, and HD95 (SciPy EDT)
│   └── unet_attention.py           # Attention U-Net-Lite with GroupNorm (G=8)
├── results/
│   └── real_lits_fair_comparison.json # Verified empirical benchmark artifacts
├── scripts/
│   └── generate_review2_ppt.py     # Institutional PowerPoint deck generator
├── test_slices/                    # Demonstration CT slices for offline triage
├── tests/                          # 57 automated regression test suites
├── requirements.txt                # Complete pip dependencies
├── run.py                          # Cross-platform application entrypoint
├── start.bat                       # Windows one-click startup script
├── start.ps1                       # PowerShell startup script
└── FedLiverNet_Second_Review_Presentation.pptx # Official 19-slide presentation
```

---

## 📜 Regulatory Notice & Disclaimer
*This platform is an investigational AI-assisted screening decision support prototype developed for academic research purposes. It is not approved or certified as an autonomous medical diagnostic device.*
