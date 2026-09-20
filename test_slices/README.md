# Clinical Patient CT Slices for Telemedicine Workstation Testing

This directory contains 3 prepared abdominal axial CT slice images extracted and standardized from the **LiTS / FedLiverNet** clinical cohort. They are pre-formatted for direct drag-and-drop or file upload into the **FedLiverNet Telemedicine Platform** web interface (`http://127.0.0.1:8000`).

---

## 1. Provided Test Slices Overview

| File | Clinical Case | Organ / Pathology | Expected AI Prediction |
| :--- | :--- | :--- | :--- |
| **`patient_01_focal_hcc.png`** | Patient #104 | Early-Stage Focal HCC | **CRITICAL ALERT**: Focal tumor detected (~0.4% lesion area, circumscribed nodule). |
| **`patient_02_macro_hcc.png`** | Patient #078 | Infiltrative Macro-HCC | **CRITICAL ALERT**: Large hypervascular neoplasm (~3.7% lesion area, extensive infiltration). |
| **`patient_03_healthy_control.png`** | Patient #012 | Healthy Liver Control | **NORMAL VERDICT**: Homogeneous parenchyma, 0.0% tumor detected, no focal neoplastic nodule. |

---

## 2. How to Upload & Test on the Platform

1. Ensure the platform server is running (`.\start.ps1` or `python run.py`).
2. Open your web browser to **`http://127.0.0.1:8000`**.
3. Under the **Abdominal CT Screening Station** card, locate the upload box:
   > *"Or upload patient CT slice (PNG, JPG) to test edge inference"*
4. Either:
   - **Click the box** to open the file picker and select one of the PNG files from `test_slices/`.
   - **Drag and drop** any of the 3 images directly into the upload area.
5. The edge inference engine will immediately process the image in real time (<150 ms) and display:
   - The dual-canvas CT slice with predicted transparent segmentation mask (Green = Liver Parenchyma, Red = Focal HCC Lesion).
   - Real-time quantitative volumetric metrics (Tumor Area %, Parenchymal ratio, Confidence score).
   - Diagnostic verdict banner and suggested radiologist review notes.

---

## 3. Where to Get Additional Real-World CT Scans

If you want to test more clinical CT scans or download full 3D patient volumes:

1. **LiTS (Liver Tumor Segmentation Benchmark)**:
   - [Kaggle LiTS Dataset](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation) - 130 contrast-enhanced 3D abdominal CT scans with expert consensus segmentations.
2. **TCIA (The Cancer Imaging Archive - TCGA-LIHC)**:
   - [TCGA-LIHC Collection](https://www.cancerimagingarchive.net/collection/tcga-lihc/) - Public DICOM repository of clinical liver hepatocellular carcinoma CT series.
3. **CHAOS Challenge (Combined Healthy Abdominal Organ Segmentation)**:
   - [CHAOS Grand Challenge](https://chaos.grand-challenge.org/) - Healthy abdominal CT series covering liver, spleen, and kidneys.
4. **Radiopaedia Open Cases**:
   - [Radiopaedia.org Hepatocellular Carcinoma](https://radiopaedia.org/articles/hepatocellular-carcinoma) - Axial CT slice image galleries and diagnostic teaching cases.
