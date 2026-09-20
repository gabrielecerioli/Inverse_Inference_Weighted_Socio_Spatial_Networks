"""
Exact Spatio-Social Parameter Inference for Weighted Networks.

Implements the deterministic, physics-informed inference pipeline formulated
in Cerioli & Cerioli (2026), Section I:
- Empirical baseline geographic interaction scale r_0 (Eq. 5).
- Physics-informed spectral residual initialization on R_init (Section I.B, Eq. 6).
- Poisson Negative Log-Likelihood with Tikhonov regularization gamma = 0.25 (Eq. 5).
- Two-stage hybrid optimization: Adam warm-up (eta = 0.04, T = 40) + quasi-Newton
  L-BFGS refinement with strong Wolfe line search (Section I.C, Appendix B).
- Profile likelihood estimation for Fisher-curvature confidence intervals on lambda
  (Section I.D, Eqs. 7-9).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist
import torch


def compute_empirical_r0(D_phys_np: np.ndarray) -> float:
    """
    Compute baseline interaction scale r_0 = (2 / N) * sum_i min_{j != i} D_ij^phys
    (twice the average nearest-neighbor geographic distance, Section I.B, Eq. 5).
    """
    D_copy = np.copy(D_phys_np)
    np.fill_diagonal(D_copy, np.inf)
    nn_distances = np.min(D_copy, axis=1)
    r0 = float(2.0 * np.mean(nn_distances))
    return max(1e-4, r0)


def spectral_residual_initialization(
    D_phys_np: np.ndarray,
    W_obs: np.ndarray,
    r0: float,
    mean_strength: float,
    soc_dim: int = 2,
    sigma_gauge: float = 1.0,
) -> np.ndarray:
    """
    Physics-informed spectral decomposition of excess non-spatial connectivity
    (Section I.B, Steps 1-4, Eq. 6):
    1. Pure geographic baseline W_0^sp from K_0^sp = exp(-D_phys / r0).
    2. Residual extraction: R_init = max(0, W_obs - W_0^sp).
    3. Truncated SVD: S^(0) = U_ds @ Sigma_ds^(1/2).
    4. Gauge calibration: standardize each latent dimension to zero mean and unit variance.
    """
    N = len(W_obs)
    # 1. Geographic baseline
    K_sp0 = np.exp(-D_phys_np / r0)
    np.fill_diagonal(K_sp0, 0.0)
    row_sums = K_sp0.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1e-12, row_sums)
    W_tilde_sp0 = 0.5 * (K_sp0 / row_sums + K_sp0 / row_sums.T)
    W_sp0 = mean_strength * W_tilde_sp0

    # 2. Non-spatial residual extraction
    R_init = np.maximum(0.0, W_obs - W_sp0)
    np.fill_diagonal(R_init, 0.0)

    # 3. Low-rank spectral embedding (Eckart-Young-Mirsky theorem, Eq. 6)
    try:
        U, sv, _ = np.linalg.svd(R_init, full_matrices=False)
        S_init = U[:, :soc_dim] * np.sqrt(np.maximum(0.0, sv[:soc_dim]))
    except Exception:
        S_init = np.random.randn(N, soc_dim).astype(np.float32) * 0.1

    # 4. Gauge calibration (zero mean and unit variance per latent dimension)
    S_init = S_init - S_init.mean(axis=0, keepdims=True)
    stds = S_init.std(axis=0, keepdims=True)
    stds = np.where(stds > 1e-6, stds, 1.0)
    S_init = (S_init / stds) * sigma_gauge

    return S_init.astype(np.float32)


def infer_parameters_exact(
    X_phys: np.ndarray,
    W_obs: np.ndarray,
    soc_dim: int = 2,
    sigma_gauge: float = 1.0,
    gamma_reg: float = 0.25,
    adam_lr: float = 0.04,
    adam_steps: int = 40,
    lbfgs_steps: int = 40,
    compute_profile_ci: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Union[float, np.ndarray, Tuple[float, float]]]:
    """
    Infer socio-spatial mixing parameter lambda in [0, 1], characteristic physical
    scale r > 0, and latent coordinates S via hybrid Adam + L-BFGS optimization.

    Parameters
    ----------
    X_phys : np.ndarray of shape (N, d_phys)
        Observed Euclidean coordinates of all nodes.
    W_obs : np.ndarray of shape (N, N)
        Observed symmetric weighted adjacency matrix.
    soc_dim : int, default=2
        Dimensionality of the latent social space d_s (Appendix E).
    sigma_gauge : float, default=1.0
        Arbitrary gauge scale fixing the metric unit of social space (Section I.A).
    gamma_reg : float, default=0.25
        Tikhonov regularizer weight on (ln r - ln r_0)^2 (Section I.B, Eq. 5).
    adam_lr : float, default=0.04
        Adam learning rate eta_Adam (Section I.C, Appendix B.1).
    adam_steps : int, default=40
        Warm-up exploration budget T_Adam (Appendix B.1).
    lbfgs_steps : int, default=40
        Quasi-Newton refinement budget T_L-BFGS (Appendix B.2).
    compute_profile_ci : bool, default=False
        Whether to compute Fisher-curvature profile likelihood 95% CI on lambda
        (Section I.D, Eqs. 7-9).
    device : str or torch.device, optional
        Computation device ('cpu' or 'cuda').
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        - 'lambda': inferred mixing parameter in [0, 1].
        - 'r': inferred characteristic physical scale.
        - 'S_soc': reconstructed latent social coordinates, shape (N, soc_dim).
        - 'loss': minimized Poisson NLL objective value.
        - 'r0': empirical baseline physical scale.
        - 'ci_95_lambda': (lower, upper) confidence interval (if compute_profile_ci=True).
        - 'sigma_lambda': asymptotic standard error on lambda (if compute_profile_ci=True).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    N = len(X_phys)
    if W_obs.shape != (N, N):
        raise ValueError(f"Matrix shape mismatch: X has {N} rows, W has shape {W_obs.shape}")

    # Empirical average nodal strength <s> = (1 / N) * sum_{i,j} W_ij^obs (Eq. 4)
    mean_strength = float(W_obs.sum(axis=1).mean())
    if mean_strength <= 1e-9:
        mean_strength = 1e-3

    # Pairwise physical Euclidean distance matrix
    D_phys_np = cdist(X_phys, X_phys)
    r0 = compute_empirical_r0(D_phys_np)
    log_r0 = float(np.log(r0))

    # Physics-informed spectral residual initialization (Section I.B)
    S_init = spectral_residual_initialization(
        D_phys_np, W_obs, r0, mean_strength, soc_dim=soc_dim, sigma_gauge=sigma_gauge
    )

    # Tensor pre-allocation on device
    iu, ju = np.triu_indices(N, k=1)
    D_phys_t = torch.tensor(D_phys_np, dtype=torch.float32, device=device)
    W_obs_u = torch.tensor(W_obs[iu, ju], dtype=torch.float32, device=device)
    iu_t = torch.tensor(iu, dtype=torch.long, device=device)
    ju_t = torch.tensor(ju, dtype=torch.long, device=device)
    eye_mask = 1.0 - torch.eye(N, dtype=torch.float32, device=device)

    # Trainable parameters: logit(lambda), ln(r), and latent coordinates S
    logit_lam = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    log_r = torch.tensor(log_r0, dtype=torch.float32, requires_grad=True, device=device)
    S_param = torch.tensor(S_init, dtype=torch.float32, requires_grad=True, device=device)

    def evaluate_loss(fixed_lam=None):
        lam = fixed_lam if fixed_lam is not None else torch.sigmoid(logit_lam)
        r = torch.exp(torch.clamp(log_r, log_r0 - 3.5, log_r0 + 3.5))

        # Spatial kernel (Eq. 1)
        K_sp = torch.exp(-D_phys_t / r) * eye_mask

        # Social kernel (Eq. 1)
        diff = S_param.unsqueeze(1) - S_param.unsqueeze(0)
        dist_soc = torch.sqrt((diff ** 2).sum(-1) + 1e-12) + 1e9 * (1.0 - eye_mask)
        K_soc = torch.exp(-dist_soc / sigma_gauge) * eye_mask

        # Convex combination of kernels (Eq. 2)
        W_raw = (1.0 - lam) * K_sp + lam * K_soc
        row_sums = W_raw.sum(dim=1, keepdim=True) + 1e-12

        # Doubly-balanced connection propensity matrix W_tilde (Eq. 3)
        # Expected link intensity W_pred = <s> * W_tilde (Eq. 4)
        W_pred = 0.5 * mean_strength * (W_raw / row_sums + W_raw / row_sums.t())

        wu = W_pred[iu_t, ju_t]
        # Poisson NLL + Tikhonov regularizer (gamma / 2) * (ln r - ln r0)^2 (Eq. 5)
        nll = (wu - W_obs_u * torch.log(wu + 1e-12)).sum()
        reg = 0.5 * gamma_reg * (log_r - log_r0) ** 2
        return nll + reg

    # Phase A: Adam Stochastic Exploration (Section I.C, Appendix B.1)
    if adam_steps > 0:
        opt_adam = torch.optim.Adam([
            {"params": [logit_lam], "lr": adam_lr},
            {"params": [log_r], "lr": adam_lr},
            {"params": [S_param], "lr": adam_lr},
        ])
        for _ in range(adam_steps):
            opt_adam.zero_grad(set_to_none=True)
            loss_val = evaluate_loss()
            loss_val.backward()
            opt_adam.step()

    # Phase B: Quasi-Newton L-BFGS Refinement (Section I.C, Appendix B.2)
    if lbfgs_steps > 0:
        opt_lbfgs = torch.optim.LBFGS(
            [logit_lam, log_r, S_param],
            lr=0.5,
            max_iter=lbfgs_steps,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_lbfgs.zero_grad(set_to_none=True)
            loss_val = evaluate_loss()
            loss_val.backward()
            return loss_val

        try:
            opt_lbfgs.step(closure)
        except Exception:
            pass

    with torch.no_grad():
        final_lam = float(torch.sigmoid(logit_lam).detach().cpu().item())
        final_r = float(torch.exp(torch.clamp(log_r, log_r0 - 3.5, log_r0 + 3.5)).detach().cpu().item())
        final_loss = float(evaluate_loss().detach().cpu().item())
        final_S = S_param.detach().cpu().numpy()

    result = {
        "lambda": final_lam,
        "r": final_r,
        "S_soc": final_S,
        "loss": final_loss,
        "r0": r0,
    }

    # Optional: Profile Likelihood Uncertainty Analysis (Section I.D, Eqs. 7-9)
    if compute_profile_ci:
        delta = 0.04
        lams_eval = [
            max(1e-4, final_lam - delta),
            min(1.0 - 1e-4, final_lam + delta),
        ]
        l_profs = []

        for lam_p in lams_eval:
            # Re-optimize conditioned on fixed lambda
            t_lam = torch.tensor(lam_p, dtype=torch.float32, device=device)
            p_log_r = torch.tensor(float(np.log(final_r)), dtype=torch.float32, requires_grad=True, device=device)
            p_S = torch.tensor(final_S, dtype=torch.float32, requires_grad=True, device=device)

            p_opt = torch.optim.LBFGS([p_log_r, p_S], lr=0.5, max_iter=25, line_search_fn="strong_wolfe")

            def p_closure():
                p_opt.zero_grad(set_to_none=True)
                r_val = torch.exp(torch.clamp(p_log_r, log_r0 - 3.5, log_r0 + 3.5))
                k_sp = torch.exp(-D_phys_t / r_val) * eye_mask
                d_s = torch.sqrt(((p_S.unsqueeze(1) - p_S.unsqueeze(0)) ** 2).sum(-1) + 1e-12) + 1e9 * (1.0 - eye_mask)
                k_sc = torch.exp(-d_s / sigma_gauge) * eye_mask
                w_r = (1.0 - t_lam) * k_sp + t_lam * k_sc
                r_s = w_r.sum(dim=1, keepdim=True) + 1e-12
                w_p = 0.5 * mean_strength * (w_r / r_s + w_r / r_s.t())
                wu_p = w_p[iu_t, ju_t]
                l_val = (wu_p - W_obs_u * torch.log(wu_p + 1e-12)).sum() + 0.5 * gamma_reg * (p_log_r - log_r0) ** 2
                l_val.backward()
                return l_val

            try:
                p_opt.step(p_closure)
            except Exception:
                pass

            l_profs.append(float(p_closure().detach().cpu().item()))

        # Empirical Fisher curvature d_NLL^2 (Eq. 8)
        d2_nll = (l_profs[1] - 2.0 * final_loss + l_profs[0]) / (delta ** 2)
        if d2_nll > 0:
            sigma_lam = float(1.0 / np.sqrt(d2_nll))
            ci_low = max(0.0, final_lam - 1.96 * sigma_lam)
            ci_high = min(1.0, final_lam + 1.96 * sigma_lam)
        else:
            sigma_lam = np.nan
            ci_low, ci_high = np.nan, np.nan

        result["sigma_lambda"] = sigma_lam
        result["ci_95_lambda"] = (ci_low, ci_high)

    return result


if __name__ == "__main__":
    from spatiosocial_network_generator import (
        generate_spatiosocial_network,
        latent_metric_deficit,
        orthogonal_procrustes_alignment,
    )

    print("=" * 70)
    print("DEMO: Exact Inference on Synthetic Benchmark Network")
    print("=" * 70)

    true_lam = 0.55
    true_r = 0.20
    true_sigma = 0.15

    net = generate_spatiosocial_network(
        n_nodes=150,
        lam=true_lam,
        r=true_r,
        sigma=true_sigma,
        n_gaussians=3,
        seed=42,
    )

    result = infer_parameters_exact(
        net["X_phys"],
        net["W"],
        compute_profile_ci=True,
        seed=42,
    )

    rho, D_soc = latent_metric_deficit(net["S_soc"], result["S_soc"])
    S_aligned, _, _ = orthogonal_procrustes_alignment(net["S_soc"], result["S_soc"])

    print(f"Ground Truth : lambda = {true_lam:.3f} | r = {true_r:.3f}")
    print(f"Inferred     : lambda = {result['lambda']:.3f} | r = {result['r']:.3f}")
    print(f"95% CI lambda: [{result['ci_95_lambda'][0]:.3f}, {result['ci_95_lambda'][1]:.3f}] (sigma = {result['sigma_lambda']:.4f})")
    print(f"Social Metric: Pearson distance correlation rho = {rho:.3f} | Deficit D_soc = {D_soc:.3f}")
    print(f"Baseline r0  : {result['r0']:.4f}")
    print(f"Final NLL    : {result['loss']:.2f}")
