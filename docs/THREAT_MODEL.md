# FedLiverNet: Threat Model & Security Architecture

**Document Purpose**: Formal security boundary specification, trust assumptions, and threat mitigations for the *FedLiverNet* Telemedicine Platform.  
**Academic Target**: Final Year Project Committee Review & External Viva Defense  
**Audience**: AI/ML Engineers, Information Security Officers, Clinical Governance Reviewers

---

## 1. System Overview & Trust Assumptions

In a multi-institutional medical federated learning network, three primary entities interact:

```
[ Primary Care Clinic A ]     [ Diagnostic Outpost B ]     [ Oncology Hub C ]
(Local Abdominal CT Scans)   (Local Abdominal CT Scans)   (Local Abdominal CT Scans)
            │                            │                            │
            ▼                            ▼                            ▼
      [ Client A ]                 [ Client B ]                 [ Client C ]
            │                            │                            │
            └────────────────────────────┼────────────────────────────┘
                                         ▼
                         [ Central Aggregation Coordinator ]
                                         │
                                         ▼
                             [ Global / Cluster Model ]
```

### Trust Boundary Axioms:
1. **Raw Patient Imagery Never Leaves Client Premises**: Raw DICOM/NIfTI scans, segmentation contours, and patient clinical identifiers are stored strictly within the hospital edge storage boundary.
2. **Untrusted Communication Channels**: Communication between client nodes and the coordinator traverses public or untrusted broadband/cellular uplinks (secured via standard TLS 1.3 in transport).
3. **Semi-Honest (Honest-but-Curious) Central Coordinator**: The central server executes the defined aggregation protocol faithfully according to the algorithm, but actively inspects received client payloads in an attempt to infer private patient pathology or reconstruct CT scans.

---

## 2. Threat Analysis & Mitigations

| Threat Vector | Adversary Profile | Attack Mechanism | Implemented Defense Mechanism | Residual Risk / Limitation |
| :--- | :--- | :--- | :--- | :--- |
| **T1: Deep Gradient Inversion Attack** | Honest-but-Curious Central Server | Server optimizes dummy input $\hat{x}$ to match observed client update $\Delta w$ (Zhu et al., 2019), attempting to reconstruct abdominal CT anatomy. | **Client-Side Differential Privacy (DP-SGD)**: $L_2$ global norm clipping ($C=1.0$) bounds update sensitivity; calibrated zero-mean Gaussian noise $\mathcal{N}(0, \sigma^2 \mathbf{I})$ perturbs gradients. | Higher noise ($\sigma$) reduces subtle lesion contour Dice. |
| **T2: Single-Client Update Snooping** | Semi-Honest Server or Network Eavesdropper | Inspecting individual hospital model updates to deduce whether a clinic treats high-stage liver cancer vs. benign cysts. | **Zero-Sum Additive Masking**: Client pairs $(i, j)$ generate symmetric random masks ($M_{i,j} = -M_{j,i}$). The server receives only obscured parameter sums: $\sum \tilde{w}_i = \sum w_i + 0$. | Assumes all participating clients complete upload (see Section 3). |
| **T3: Intra-Patient Data Leakage** | Machine Learning Pipeline Flaw | Slices from the same patient appearing across both training and test sets, artificially inflating validation Dice scores. | **Patient-Level Partitioning (`data/patient_splitter.py`)**: Slices are grouped strictly by `subject_id`. Entire patients are isolated into Train (70%), Val (15%), or Test (15%). | None; 0% patient overlap mathematically enforced. |
| **T4: Offline Link Sniffing & Tampering** | Local Network Eavesdropper | Intercepting queued diagnostic reports during network outages in rural outposts. | **Embedded SQLite Storage (WAL Mode)**: Local files stored with OS-level permissions; offline sync payloads serialized and replayed with idempotent session tokens. | Local device physical security is assumed. |
| **T5: Poisoning / Byzantine Clients** | Malicious Compromised Client | Injecting adversarial gradients to degrade global tumor sensitivity. | *Out of Scope for Current Review*: Addressed in future work via robust aggregation (Krum, Bulyan, or coordinate-wise median). | Documented as honest boundary. |

---

## 3. Cryptographic Limitations & Dropout Behavior (Viva Talking Points)

> [!IMPORTANT]
> **Academic Integrity Statement on Dropout Handling**:  
> In your project viva and examination, state this limitation transparently:
> 
> *Our simplified Zero-Sum Additive Masking protocol assumes that all $K$ clients that generate pairwise masks complete their upload for that communication round.*
> 
> - **If all $K$ clients participate**: $\sum_{i=1}^K \sum_{j \neq i} M_{i,j} = \mathbf{0}$, perfectly canceling out all random masks.
> - **If client $k$ drops offline mid-round**: The remaining $K-1$ clients' masks will **not** sum to zero ($\sum_{i \neq k} M_{i,j} \neq \mathbf{0}$).
> - **Implemented Safe Behavior**: The coordinator detects that the mask checksum does not balance, **safely aborts the round, and triggers a round restart**. It never silently computes an incorrect parameter aggregate.
> - **Future Extension**: Full 3-phase threshold secret sharing (Bonawitz et al., Shamir's Secret Sharing) to reconstruct masks of dropped clients without restarting the round.

---

## 4. Privacy Accountant & Formal Bound

The system implements Client-Level $(\epsilon, \delta)$-Differential Privacy tracking across $R$ rounds:
$$\sigma = \frac{C \sqrt{2 \ln(1.25 / \delta)}}{\epsilon}$$

For $C = 1.0$, $\delta = 10^{-5}$, and balanced budget $\epsilon = 1.0$:
- Calibrated standard deviation: $\sigma \approx 4.8448$
- Analytical composition across $R$ rounds is tracked via `PrivacyAccountant` in [`privacy/dp_mechanism.py`](file:///e:/Projects/telemedicine%20platform/privacy/dp_mechanism.py).
