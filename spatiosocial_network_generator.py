"""
Spatio-Social Synthetic Network Generator.

Generates synthetic networks governed by a dual physical-social latent space
as formulated in Cerioli & Cerioli (2026), Appendix A.1:
- Physical coordinates x_i in [0, 1]^2 sampled uniformly.
- Latent social coordinates s_i in [0, 1]^ds sampled from a Gaussian Mixture Model (GMM).
- Dual-kernel connectivity: W_raw = (1 - lambda) * K_sp + lambda * K_soc.
- Doubly-balanced connection propensity W_tilde and Poisson link sampling:
  W_obs ~ Poisson(<s > * W_tilde), with <s > = N in benchmark ensembles.
- Metric evaluation via Orthogonal Procrustes alignment (Appendix A.2).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist


def sample_social_gmm(
    n_nodes: int,
    n_gaussians: Optional[int] = None,
    dim: int = 2,
    domain_size: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample latent social coordinates from a Gaussian Mixture Model (GMM)
    according to Appendix A.1.b (Eq. A2).

    Parameters
    ----------
    n_nodes : int
        Number of nodes N.
    n_gaussians : int, optional
        Number of mixture components K. If None, sampled uniformly from {2, ..., 8}.
    dim : int, default=2
        Dimensionality of the latent social space d_s.
    domain_size : float, default=1.0
        Bounding hypercube size.
    rng : np.random.Generator, optional
        NumPy random number generator.

    Returns
    -------
    coords : np.ndarray of shape (n_nodes, dim)
        Min-max normalized latent social coordinates strictly in [0, domain_size]^dim.
    assignments : np.ndarray of shape (n_nodes,)
        Cluster assignments for each node.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_gaussians is None:
        n_gaussians = int(rng.integers(2, 9))

    # Cluster mixing weights: pi ~ Dirichlet(1_K)
    weights = rng.dirichlet(np.ones(n_gaussians))

    # Cluster centers: mu_k ~ U([0.1, 0.9]^ds) * domain_size
    centers = rng.uniform(0.1 * domain_size, 0.9 * domain_size, size=(n_gaussians, dim))

    # Diagonal covariances: sigma_{k, d}^2 ~ U(0.01, 0.05) * domain_size^2
    variances = rng.uniform(0.01, 0.05, size=(n_gaussians, dim)) * (domain_size ** 2)
    stds = np.sqrt(variances)

    assignments = rng.choice(n_gaussians, size=n_nodes, p=weights)
    coords = np.zeros((n_nodes, dim), dtype=float)

    for k in range(n_gaussians):
        mask = (assignments == k)
        count = int(np.sum(mask))
        if count > 0:
            coords[mask] = rng.normal(loc=centers[k], scale=stds[k], size=(count, dim))

    # Coordinate-wise min-max normalization to reside strictly within [0, domain_size]^ds (Eq. A2)
    c_min = coords.min(axis=0)
    c_max = coords.max(axis=0)
    scale = np.where(c_max - c_min > 1e-9, c_max - c_min, 1.0)
    coords = (coords - c_min) / scale * domain_size

    return coords, assignments


def generate_spatiosocial_network(
    n_nodes: int = 200,
    lam: Optional[float] = None,
    r: Optional[float] = None,
    sigma: Optional[float] = None,
    n_gaussians: Optional[int] = None,
    avg_strength: Optional[float] = None,
    phys_dim: int = 2,
    soc_dim: int = 2,
    domain_size: float = 1.0,
    graph_type: str = "poisson",
    seed: Optional[int] = None,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """
    Generate a synthetic spatio-social network instance (Appendix A.1).

    Parameters
    ----------
    n_nodes : int, default=200
        Network size N.
    lam : float, optional
        Socio-spatial mixing parameter in [0, 1]. If None, drawn from U(0, 1).
    r : float, optional
        Physical interaction scale. If None, ln(r) ~ U(ln 0.03, ln 0.50).
    sigma : float, optional
        Social interaction scale. If None, ln(sigma) ~ U(ln 0.03, ln 0.50).
    n_gaussians : int, optional
        Number of GMM clusters in social space. If None, sampled from {2, ..., 8}.
    avg_strength : float, optional
        Target average nodal strength <s>. In synthetic benchmarks (Eq. A5), <s> = N.
        Defaults to float(n_nodes) if None.
    phys_dim : int, default=2
        Dimensionality of physical space.
    soc_dim : int, default=2
        Dimensionality of latent social space d_s.
    domain_size : float, default=1.0
        Domain side length for coordinates.
    graph_type : {'poisson', 'weighted', 'bernoulli'}, default='poisson'
        - 'poisson': discrete counts sampled from Poisson(W_pred) (Eq. A5).
        - 'weighted': continuous propensity matrix W_pred (Eq. 4).
        - 'bernoulli': binary adjacency with link probabilities 1 - exp(-W_pred).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        - 'X_phys': physical coordinates, shape (N, phys_dim).
        - 'S_soc': latent social coordinates, shape (N, soc_dim).
        - 'social_clusters': GMM component indices, shape (N,).
        - 'D_phys': physical distance matrix, shape (N, N).
        - 'D_soc': latent social distance matrix, shape (N, N).
        - 'W': observed adjacency/weight matrix W_obs, shape (N, N).
        - 'W_pred': expected link intensity matrix W_pred, shape (N, N).
        - 'W_tilde': doubly-balanced connection propensity matrix, shape (N, N).
        - 'parameters': ground-truth parameter dictionary.
    """
    rng = np.random.default_rng(seed)

    if lam is None:
        lam = float(rng.uniform(0.0, 1.0))
    if r is None:
        r = float(np.exp(rng.uniform(np.log(0.03), np.log(0.50)))) * domain_size
    if sigma is None:
        sigma = float(np.exp(rng.uniform(np.log(0.03), np.log(0.50)))) * domain_size
    if avg_strength is None:
        avg_strength = float(n_nodes)

    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if r <= 0 or sigma <= 0:
        raise ValueError(f"Scales r and sigma must be positive, got r={r}, sigma={sigma}")

    # 1. Physical coordinates x_i ~ U([0, domain_size]^phys_dim) (Appendix A.1.b)
    X_phys = rng.uniform(0.0, domain_size, size=(n_nodes, phys_dim))

    # 2. Latent social coordinates S_true from GMM in [0, domain_size]^soc_dim (Eq. A2)
    S_soc, cluster_ids = sample_social_gmm(
        n_nodes=n_nodes,
        n_gaussians=n_gaussians,
        dim=soc_dim,
        domain_size=domain_size,
        rng=rng,
    )

    # 3. Pairwise Euclidean distances
    D_phys = cdist(X_phys, X_phys)
    D_soc = cdist(S_soc, S_soc)

    # 4. Dual-kernel affinity: W_raw_ij = (1 - lam)*K_sp + lam*K_soc (Eq. A3)
    eye = np.eye(n_nodes, dtype=bool)
    K_sp = np.exp(-D_phys / r)
    K_soc = np.exp(-D_soc / sigma)
    K_sp[eye] = 0.0
    K_soc[eye] = 0.0

    W_raw = (1.0 - lam) * K_sp + lam * K_soc
    W_raw[eye] = 0.0

    # 5. Doubly-balanced row-normalization (Eq. A4)
    row_sums = W_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1e-12, row_sums)
    W_tilde = 0.5 * (W_raw / row_sums + W_raw / row_sums.T)
    W_tilde[eye] = 0.0

    # 6. Expected Poisson link intensity matrix (Eq. 4 & Eq. A5)
    W_pred = avg_strength * W_tilde
    W_pred[eye] = 0.0

    # 7. Sampling discrete edge counts
    iu, ju = np.triu_indices(n_nodes, k=1)
    rates = W_pred[iu, ju]

    if graph_type == "poisson":
        w_vals = rng.poisson(rates).astype(float)
        W = np.zeros((n_nodes, n_nodes), dtype=float)
        W[iu, ju] = w_vals
        W[ju, iu] = w_vals
    elif graph_type == "weighted":
        W = np.copy(W_pred)
    elif graph_type == "bernoulli":
        probs = 1.0 - np.exp(-rates)
        edge_vals = (rng.uniform(size=len(rates)) < probs).astype(float)
        W = np.zeros((n_nodes, n_nodes), dtype=float)
        W[iu, ju] = edge_vals
        W[ju, iu] = edge_vals
    else:
        raise ValueError(f"Unknown graph_type '{graph_type}'. Choose 'poisson', 'weighted', or 'bernoulli'.")

    return {
        "X_phys": X_phys,
        "S_soc": S_soc,
        "social_clusters": cluster_ids,
        "D_phys": D_phys,
        "D_soc": D_soc,
        "W": W,
        "W_pred": W_pred,
        "W_tilde": W_tilde,
        "parameters": {
            "n_nodes": n_nodes,
            "lam": lam,
            "r": r,
            "sigma": sigma,
            "avg_strength": avg_strength,
            "graph_type": graph_type,
            "seed": seed,
        },
    }


def orthogonal_procrustes_alignment(
    S_true: np.ndarray,
    S_hat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the constrained least-squares Orthogonal Procrustes problem with an
    isotropic scale factor delta > 0, as derived in Appendix A.2 (Eqs. A6-A10):
        min_{R in O(d_s), delta > 0} || S_true_bar - delta * S_hat_bar * R ||_F^2

    Parameters
    ----------
    S_true : np.ndarray of shape (N, d_s)
        Ground-truth latent coordinates.
    S_hat : np.ndarray of shape (N, d_s)
        Inferred latent coordinates.

    Returns
    -------
    S_aligned : np.ndarray of shape (N, d_s)
        Aligned coordinates: S_aligned = delta * S_hat_bar * R.
    R_opt : np.ndarray of shape (d_s, d_s)
        Optimal orthogonal rotation/reflection matrix.
    delta_opt : float
        Optimal isotropic scaling factor.
    """
    N, d_s = S_true.shape
    # Center both coordinate matrices (Eq. A6)
    S_true_bar = S_true - S_true.mean(axis=0, keepdims=True)
    S_hat_bar = S_hat - S_hat.mean(axis=0, keepdims=True)

    # Cross-covariance dispersion matrix M = S_hat_bar^T @ S_true_bar (Eq. A7)
    M = S_hat_bar.T @ S_true_bar

    # SVD of cross-dispersion matrix: M = U @ diag(sigma) @ V^T
    U, sv, Vt = np.linalg.svd(M)

    # Optimal orthogonal transformation R* = U @ V^T (Eq. A9)
    R_opt = U @ Vt

    # Optimal isotropic scaling factor delta* = Tr(Sigma) / ||S_hat_bar||_F^2 (Eq. A10)
    norm_sq = float(np.sum(S_hat_bar ** 2))
    delta_opt = float(np.sum(sv) / (norm_sq + 1e-12))

    S_aligned = delta_opt * (S_hat_bar @ R_opt)
    return S_aligned, R_opt, delta_opt


def latent_metric_deficit(
    S_true: np.ndarray,
    S_hat: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute the Pearson distance correlation rho_soc and latent metric deficit
    D_soc = 1 - rho_soc across unique pairwise Euclidean distances (Eq. A11).

    Returns
    -------
    rho_soc : float
        Pearson distance correlation between true and reconstructed pairwise distances.
    D_soc : float
        Latent metric space deficit: D_soc = 1 - rho_soc.
    """
    N = len(S_true)
    iu, ju = np.triu_indices(N, k=1)

    D_true_vec = cdist(S_true, S_true)[iu, ju]
    D_hat_vec = cdist(S_hat, S_hat)[iu, ju]

    # Pearson correlation
    dev_true = D_true_vec - D_true_vec.mean()
    dev_hat = D_hat_vec - D_hat_vec.mean()

    denom = np.sqrt(np.sum(dev_true ** 2) * np.sum(dev_hat ** 2))
    if denom < 1e-12:
        rho_soc = 0.0
    else:
        rho_soc = float(np.sum(dev_true * dev_hat) / denom)

    D_soc = float(1.0 - rho_soc)
    return rho_soc, D_soc


if __name__ == "__main__":
    net = generate_spatiosocial_network(
        n_nodes=200,
        lam=0.45,
        r=0.20,
        sigma=0.15,
        n_gaussians=4,
        seed=42,
    )

    W = net["W"]
    nonzeros = int((W > 0).sum() // 2)
    mean_strength = float(W.sum(axis=1).mean())

    print("Synthetic Spatio-Social Network (Appendix A.1):")
    print(f"  Nodes N: {net['parameters']['n_nodes']}")
    print(f"  Lambda: {net['parameters']['lam']:.3f}")
    print(f"  Physical scale r: {net['parameters']['r']:.3f}")
    print(f"  Social scale sigma: {net['parameters']['sigma']:.3f}")
    print(f"  Average nodal strength <s >: {mean_strength:.2f} (Target N = {net['parameters']['n_nodes']})")
    print(f"  Unique active links: {nonzeros:,}")
