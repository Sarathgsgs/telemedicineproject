# Disease Taxonomy & Clinical Screening Mapping

## 1. Ground Truth Voxel Taxonomy (LiTS Benchmark Grounding)

In medical imaging and deep learning research, maintaining strict alignment between semantic labels and dataset annotations is paramount. The primary benchmark for this research—the **Liver Tumor Segmentation (LiTS) Challenge** dataset—features three-level voxel ground truth annotations:

| Voxel Class ID | Semantic Entity | Radiological Characteristics (CT / Axial View) | Hounsfield Unit (HU) Range |
| :--- | :--- | :--- | :--- |
| **Class 0** | **Background / Non-Liver** | Air, thoracic structures, rib cage, abdominal wall, spleen, kidneys, stomach, and bowel loops. | Outside $[-100, 400]$ or non-hepatic abdominal organs. |
| **Class 1** | **Liver Parenchyma** | Homogeneous vascularized hepatic tissue forming the functional anatomical liver boundary. | Typically $+40 \text{ to } +70\text{ HU}$ on non-contrast / $+80 \text{ to } +150\text{ HU}$ on contrast-enhanced CT. |
| **Class 2** | **Liver Lesion / Tumor** | Focal neoplasms (Hepatocellular Carcinoma - HCC, secondary metastatic lesions, cysts, hemangiomas, or adenomas). | Hypodense (low attenuation relative to liver, e.g. $+20 \text{ to } +45\text{ HU}$) or hypervascular arterial phase enhancement. |

---

## 2. Scientific Integrity Note: Grounded vs. Fabricated Classes

> [!IMPORTANT]
> **Academic Integrity Statement for Review Committee**:  
> Many undergraduate projects claim multi-disease classification (e.g., "Normal vs. Cirrhosis vs. Fatty Liver vs. Hepatitis vs. Tumor") using the LiTS dataset.  
> **This is scientifically invalid.** The LiTS and CHAOS datasets do not possess pathological biopsies or clinical metadata to label diffuse liver steatosis (fatty liver) or fibrotic staging (cirrhosis); they only provide segmentations for **liver volume** and **focal tumor masses**.  
> 
> Our platform maintains strict scientific integrity by operating on **volumetric segmentation and focal lesion screening**, rather than inventing ungrounded diagnostic categories.

---

## 3. Clinical Telemedicine Screening Translation

In our telemedicine web platform, voxel-level segmentations are translated into an actionable clinical triage report for tele-radiologists and rural clinics:

```
                                  [ Input 2D Axial CT Slice ]
                                               │
                                 [ Attention U-Net-Lite ]
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
             Liver Parenchyma Area                           Lesion / Tumor Area
               (Voxel Class 1)                                 (Voxel Class 2)
                       │                                               │
                       └───────────────────────┬───────────────────────┘
                                               │
                                   [ Clinical Stratification ]
```

### Risk Stratification Criteria:
1. **Screening Tier 1 — Unremarkable / Normal Liver**:
   - Liver parenchyma detected (Class 1 area $> 500\text{ mm}^2$).
   - Zero Class 2 lesion voxels detected (or lesion confidence $< 0.15$).
   - Clinical Action: Standard routine screening follow-up.

2. **Screening Tier 2 — Suspicious Focal Lesion / Neoplasm Detected**:
   - Distinct cluster of Class 2 voxels within the hepatic envelope ($\text{Lesion Area} \ge 10\text{ mm}^2$).
   - Metrics computed:
     - **Lesion-to-Liver Volume Burden Ratio**: $\frac{\text{Area}(\text{Lesion})}{\text{Area}(\text{Liver})} \times 100\%$
     - **Mean Lesion Confidence Score**: Softmax probability across predicted lesion pixels.
     - **Estimated Lesion Axial Diameter**: Maximum Euclidean feret diameter in millimeters.
   - Clinical Action: AI-assisted triage referral for contrast-enhanced multiphase CT/MRI and specialist oncologist consult.

3. **Screening Tier 3 — Borderline / Indeterminate Scan**:
   - Isolated, noisy low-confidence clusters ($0.15 \le \text{confidence} < 0.50$, area $< 10\text{ mm}^2$).
   - Clinical Action: Flag for manual radiologist overread; recommendation for follow-up scan.

---

## 4. Regulatory & Clinical Disclaimer Framing

Every clinical report and screen within the telemedicine interface embeds the following mandatory clinical safety notice:

> **Clinical Decision Support Disclaimer**:  
> *"This system is an AI-assisted privacy-preserving screening tool intended solely to augment clinical decision-making. It is not an autonomous diagnostic device. All AI segmentation boundaries and risk metrics must be independently reviewed and verified by a licensed radiologist or healthcare professional before clinical intervention."*
