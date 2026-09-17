# Privacy Mechanisms: Mathematical Guarantees & Honest Limitations
**Module**: Phase 6 — Privacy-Preserving Federated Architecture  
**Base Paper**: *FedLiverNet*, Section 2.4 & Section 3.3  
**Audience**: Final Year Review Committee & External Viva Examiners

---

## 1. Dual-Layer Privacy Architecture

To defend against reconstructive deep leakage attacks (Zhu et al., 2019) where an adversarial server attempts to reconstruct private abdominal patient CT scans from model gradient updates, our platform incorporates a two-tier defense:

```
[ Local Client Training ]
           │
           ▼
[ Tier 1: Client-Side Differential Privacy ]
  - L2 Parameter Update Clipping: ||Delta w||_2 <= C
  - Calibrated Gaussian Mechanism: noise ~ N(0, sigma^2)
  - Analytical (epsilon, delta) Privacy Guarantee
           │
           ▼
[ Tier 2: Zero-Sum Additive Masking ]
  - Pairwise Symmetric Masks: M_{i,j} = -M_{j,i}
  - Transmitted Update: w_masked = w + sum_{j > i} M_{i,j} - sum_{j < i} M_{j,i}
           │
           ▼
[ Aggregation Server ]
  - sum_{i=1}^K w_masked_i = sum_{i=1}^K w_i + 0
  - Server never observes individual hospital weights!
```

---

## 2. Mathematical Formulations

### Differential Privacy (Gaussian Mechanism, Base Paper Eq. 7)
Given clipping bound $C$ and failure probability $\delta = 10^{-5}$, the noise variance is calibrated to privacy budget $\epsilon$:
$$\sigma = \frac{C \sqrt{2 \ln(1.25 / \delta)}}{\epsilon}$$
- **High Privacy ($\epsilon = 0.1$)**: $\sigma \approx 48.45$. High privacy guarantee, higher degradation of subtle tumor boundaries.
- **Moderate Privacy ($\epsilon = 1.0$)**: $\sigma \approx 4.84$. Balanced operating point preserving macro liver contours while bounding reconstruction attack success.
- **No Privacy ($\epsilon = \infty$)**: $\sigma = 0.0$. Raw unperturbed parameters.

### Zero-Sum Additive Masking
For $K$ clients, each client pair $(i, j)$ shares a symmetric pseudo-random mask $M_{i,j}$ generated from a shared seed. The server receives only obscured parameter vectors $\tilde{w}_i$:
$$\sum_{i=1}^K \tilde{w}_i = \sum_{i=1}^K w_i + \sum_{i=1}^K \left(\sum_{j > i} M_{i,j} - \sum_{j < i} M_{j,i}\right) = \sum_{i=1}^K w_i$$
All masks cancel to zero across the network without requiring heavy homomorphic encryption runtime overhead.

---

## 3. Honest Limitations Paragraph (Viva Defense Talking Points)

> [!IMPORTANT]
> **Academic Integrity Statement on Privacy Scope**:  
> In your project viva and final report, state these limitations transparently:
> 
> 1. **Client Population Scale**:  
>    *FedLiverNet* simulated up to 18 clients on DGX clusters. Under 3–5 clients, standard DP requires larger per-round noise ($\sigma \propto 1/\epsilon$) because noise does not average out as quickly over smaller cohorts ($1/\sqrt{K}$).
> 
> 2. **Simplified Additive Masking vs. 3-Phase Secure Aggregation**:  
>    Our additive masking provides information-theoretic privacy against an **honest-but-curious server**. However, unlike Bonawitz et al.'s full 3-phase threshold protocol (which uses Shamir's Secret Sharing to recover dropped clients), our simplified scheme assumes all participating clients in a round complete their upload. If a client drops offline mid-round, the remaining masks will not cancel to zero, requiring a round restart.
> 
> 3. **Screening vs. Diagnostic Framing**:  
>    Differential Privacy intentionally perturbs parameters. For gross organ localization (Liver Dice), the model remains highly resilient; for minute micro-metastases ($<5\text{ mm}$), excessive DP noise reduces sensitivity. This mathematically reinforces why our platform is framed as a **telemedicine screening assistance tool** rather than an autonomous diagnostic device.
