# Inferring Latent Geometries in Weighted Spatio-Functional Networks

Reference computational implementation for the statistical generative framework presented in:
> **Gabriele Cerioli and Adamo Cerioli (2026)**. *Inferring Latent Geometries in Weighted Spatio-Functional Networks*.

This repository provides reproducible implementations to generate synthetic spatio-social networks and to infer the socio-spatial mixing parameter $\lambda \in [0, 1]$, the characteristic physical interaction scale $r > 0$, and the latent social embedding coordinates $S \in \mathbb{R}^{N \times d_s}$.

---

## 📁 Repository Structure

- [`spatiosocial_network_generator.py`](spatiosocial_network_generator.py):
  - Generates synthetic networks based on the generative model (Section I.A, Appendix A.1).
  - Latent coordinates drawn from a Gaussian Mixture Model (GMM) with coordinate-wise min-max normalization into $[0, 1]^{d_s}$ (Eq. A2).
  - Convex combination of exponential spatial and social kernels (Eq. A3) followed by doubly-balanced row normalization (Eq. A4) and Poisson link sampling (Eq. A5).
  - Includes Orthogonal Procrustes alignment (Eqs. A6–A10) and latent metric deficit calculation $D_{soc} = 1 - \rho_{soc}$ (Eq. A11).

- [`exact_inference.py`](exact_inference.py):
  - Exact maximum likelihood parameter inference on the complete network (Section I.B–I.D).
  - Empirical baseline spatial scale $r_0 = \frac{2}{N}\sum_i \min_{j \neq i} D_{ij}^{phys}$ (Eq. 5).
  - Physics-informed spectral residual initialization $R_{init} = \max(0, W^{obs} - W_0^{sp})$ with truncated SVD and gauge calibration (Section I.B, Eq. 6).
  - Two-stage hybrid optimization: Adam exploration ($\eta = 0.04$, $T = 40$) followed by quasi-Newton L-BFGS refinement with strong Wolfe line search (Section I.C, Appendix B).
  - Profile likelihood estimation for calibrated Fisher-curvature 95% confidence intervals on $\lambda$ (Section I.D, Eqs. 7–9).

- [`subsampled_inference.py`](subsampled_inference.py):
  - Scalable Monte Carlo subnetwork ensembling for large networks ($N > 1500$, Appendix G.2).
  - Solves the restricted composite likelihood on independent induced subgraphs of size $n$ (default $n = 600$, Table I).
  - Aggregates estimates across $K$ subgraphs to compute ensemble means ($\bar{\lambda}, \bar{r}$) and Monte Carlo standard errors $\sigma_{\bar{\lambda}} = \sigma_\lambda / \sqrt{K}$ (Eq. G2).

---

## 📊 Datasets & Data Availability

The empirical analyses in this work rely on open-access, publicly available datasets:

### 1. Adult Female *Drosophila melanogaster* Whole-Brain Connectome
- **Dataset:** Whole-brain connectome (FAFB v783).
- **Access Portal:** FlyWire Connectome Data Explorer (Codex) at [https://codex.flywire.ai](https://codex.flywire.ai).
- **Description:** Synaptic connectivity tables, 3D spatial coordinates of synaptic contacts, proofread neuron annotations, and neuropil mesh segmentations obtained from Codex public data releases and bulk downloads.

### 2. Continental European High-Voltage Power Grid
- **Dataset:** ENTSO-E / PyPSA-Eur transmission network model.
- **Repository:** Zenodo record at [https://zenodo.org/records/14144752](https://zenodo.org/records/14144752).
- **Description:** Georeferenced substations, AC transmission lines, and physical HVDC links representing the synchronous electrical grid of continental Europe.

### 3. Global Air Transportation Network (COVID-19 Dynamics)
- **Flight Operations Data:** OpenSky Network crowdsourced ADS-B receiver infrastructure covering monthly operations and origin–destination flight frequencies throughout the COVID-19 pandemic (2019–2022).
  - Available via Zenodo: [https://zenodo.org/records/7923702](https://zenodo.org/records/7923702).
- **Geographic Airport Metadata:** ICAO identifiers, geographic coordinates, and terrestrial geometries obtained from the open-access [OurAirports](https://ourairports.com) public database.

---

## ⚙️ Installation & Requirements

Ensure Python 3.9+ is installed along with the required scientific computing dependencies:

```bash
pip install numpy scipy torch pandas
```

---

## 🚀 Quickstart

### 1. Generate a Synthetic Spatio-Social Network
```python
from spatiosocial_network_generator import generate_spatiosocial_network

net = generate_spatiosocial_network(
    n_nodes=200,
    lam=0.45,
    r=0.20,
    sigma=0.15,
    n_gaussians=4,
    seed=42,
)

X_phys = net["X_phys"]  # Physical coordinates (N, 2)
W = net["W"]            # Observed adjacency/weight matrix (N, N)
```

### 2. Exact Inference with Profile Likelihood CI (Small/Medium Graphs)
```python
from exact_inference import infer_parameters_exact

res = infer_parameters_exact(
    X_phys, W,
    compute_profile_ci=True,
    seed=42,
)
print(f"Inferred lambda : {res['lambda']:.4f}")
print(f"Inferred scale r: {res['r']:.4f}")
print(f"95% CI lambda   : [{res['ci_95_lambda'][0]:.4f}, {res['ci_95_lambda'][1]:.4f}]")
```

### 3. Induced Subgraph Ensembling (Large-Scale Graphs)
```python
from subsampled_inference import infer_parameters_subsampled

ens = infer_parameters_subsampled(
    X_phys, W,
    subgraph_size=150,
    n_ensembles=40,
    seed=42,
    verbose=True,
)
print(f"Ensemble Mean lambda: {ens['lambda_mean']:.4f} +/- {ens['lambda_se']:.4f}")
print(f"Ensemble Mean scale r: {ens['r_mean']:.4f} +/- {ens['r_se']:.4f}")
```
