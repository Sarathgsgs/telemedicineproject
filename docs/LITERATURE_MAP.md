# Literature Map & Related Works

This document details the scientific foundation of the project, mapping the base paper (*FedLiverNet*, 2026) alongside four landmark foundational papers in medical federated learning, privacy, and deep learning architectures.

---

## Paper 1: The Base Paper
**Title**: *FedLiverNet: Privacy-Preserving Federated 3D Volumetric Liver Tumor Segmentation*  
**Authors / Venue**: *Scientific Reports* (Nature Portfolio), 2026  
- **Core Contribution**: Proposed a federated 3D hybrid TransUNet framework for multi-institutional liver and tumor segmentation across heterogeneous datasets (LiTS, CHAOS). Introduced dynamic clustered federated learning with local adaptation (CFL-LA) and a 3-phase secure aggregation masking protocol.
- **Relevance to Our Project**: Serves as our primary baseline and blueprint for clinical non-IID partitioning, privacy-utility evaluation, and loss formulation ($\mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{Focal}}$).
- **What We Do Differently**: 
  - FedLiverNet requires 48 hours on 8×A100 GPUs using 3D volumetric patches ($256 \times 256 \times 64$). 
  - We adapt the pipeline to **2D axial slices with clinical HU windowing** ($[-100, 400]$), deploy a **GroupNorm Attention U-Net-lite**, and implement **CFL-lite** (evaluating cluster cosine similarity every $N$ rounds rather than continuous per-round graph re-clustering).
  - We additionally engineer a complete **offline-first edge sync manager and telemedicine web portal**, which were absent from the original paper's offline simulation.

---

## Paper 2: Foundational Federated Learning
**Title**: *Communication-Efficient Learning of Deep Networks from Decentralized Data*  
**Authors**: H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas  
**Venue**: *AISTATS* (2017)  
- **Core Contribution**: Introduced the canonical **FederatedAveraging (FedAvg)** algorithm, demonstrating that local stochastic gradient descent (SGD) followed by coordinate-wise model parameter averaging enables decentralized model training without moving user raw data.
- **Relevance to Our Project**: Provides the baseline federated aggregation strategy that our simulation uses in Phase 4.
- **What We Do Differently**:
  - McMahan's standard FedAvg assumes independent and identically distributed (IID) or mildly skewed data; in medical imaging with distinct hospital scanners and lesion stages, FedAvg diverges or suffers high inter-client variance.
  - We use FedAvg as the baseline benchmark against which our **CFL-lite personalization** demonstrates its value.

---

## Paper 3: Federated Personalization under Non-IID Skew
**Title**: *Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization Under Non-IID Data*  
**Authors**: F. Sattler, K. R. Müller, W. Samek  
**Venue**: *IEEE Transactions on Neural Networks and Learning Systems* (TNNLS), 2020  
- **Core Contribution**: Formalized mathematical clustering of federated clients by the cosine similarity of their local parameter update vectors ($\Delta w_i$), proving that clients with congruent loss landscapes naturally cluster together, resolving non-IID gradient conflict.
- **Relevance to Our Project**: Forms the mathematical underpinning of our flagship personalization module (**CFL-lite**).
- **What We Do Differently**:
  - Sattler et al. applied CFL to general benchmarks (MNIST, CIFAR-10) using recursive bi-partitioning.
  - We adapt CFL specifically for **heterogeneous medical imaging institutions** (partitioned by lesion size and disease prevalence), combine it with an immediate post-aggregation **local fine-tuning epoch**, and evaluate it on medical segmentation metrics (Dice, IoU, inter-client $\sigma$).

---

## Paper 4: Attention Gate Architectures for Medical Segmentation
**Title**: *Attention U-Net: Learning Where to Look for the Pancreas*  
**Authors**: O. Oktay, J. Schlemper, L. Le Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, B. Glocker, D. Rueckert  
**Venue**: *Medical Imaging with Deep Learning* (MIDL), 2018  
- **Core Contribution**: Introduced soft Attention Gates (AGs) integrated into the skip connections of U-Net architectures, allowing the network to automatically learn to focus on target structures of varying shapes and sizes without explicit bounding box localization.
- **Relevance to Our Project**: Liver lesions occupy only a small fraction of an abdominal CT slice; standard convolutions suffer high false-positive rates on surrounding organs (spleen, kidneys, stomach). Attention gates suppress background tissue activations.
- **What We Do Differently**:
  - We pair Attention Gates with **Group Normalization (GroupNorm)** instead of Batch Normalization.
  - In federated learning, local client batches are small and variable ($B \in [2, 8]$). BatchNorm computes unstable running statistics across small batches, causing FL divergence; GroupNorm computes statistics across channel groups per sample, maintaining complete invariance to batch size.

---

## Paper 5: Formal Privacy and Differential Privacy
**Title**: *Deep Learning with Differential Privacy*  
**Authors**: M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang  
**Venue**: *ACM Conference on Computer and Communications Security* (CCS), 2016  
- **Core Contribution**: Developed the **Differentially Private Stochastic Gradient Descent (DP-SGD)** algorithm, introducing per-sample gradient norm clipping, calibrated Gaussian noise injection, and the Moments Accountant to strictly bound $(\epsilon, \delta)$-differential privacy.
- **Relevance to Our Project**: Federated models are vulnerable to gradient inversion attacks (e.g., reconstructive attacks recovering patient CT slices from weight updates). DP provides mathematically rigorous privacy guarantees.
- **What We Do Differently**:
  - Rather than treating privacy as a black-box constraint, we implement an empirical sweep across $\epsilon \in \{0.1, 0.5, 1.0, 5.0, \infty\}$ to systematically quantify the **Privacy-Utility Tradeoff** on medical image segmentation, mapping the exact Dice degradation curve for clinical practitioners.
