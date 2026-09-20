"""
Subsampled Ensemble Inference for Large-Scale Spatio-Social Networks.

Implements the scalable Monte Carlo subnetwork ensembling pipeline formulated
in Cerioli & Cerioli (2026), Appendix G.2 (Eqs. G1-G2):
- Evaluates induced subgraphs V_sub of size n << N (e.g. n = 600).
- Executes the two-stage hybrid protocol (physics-informed spectral initialization,
  Adam exploration, and quasi-Newton L-BFGS polish) on each replica.
- Aggregates estimates across an ensemble of K independent subgraphs (Eq. G2):
    lambda_bar = (1 / K) * sum_k lambda_hat_k
    r_bar = (1 / K) * sum_k r_hat_k
    SE(lambda_bar) = std(lambda_hat) / sqrt(K)
    SE(r_bar) = std(r_hat) / sqrt(K)
"""

from typing import Dict, Optional, Union
import time
import numpy as np
import pandas as pd
import torch

from exact_inference import infer_parameters_exact


def infer_parameters_subsampled(
    X_phys: np.ndarray,
    W_obs: np.ndarray,
    subgraph_size: int = 600,
    n_ensembles: int = 40,
    sampling_strategy: str = "uniform",
    pool_size: Optional[int] = None,
    soc_dim: int = 2,
    sigma_gauge: float = 1.0,
    gamma_reg: float = 0.25,
    adam_lr: float = 0.04,
    adam_steps: int = 40,
    lbfgs_steps: int = 40,
    device: Optional[Union[str, torch.device]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Estimate (lambda, r) on massive networks via induced subgraph Monte Carlo ensembling
    (Appendix G.2, Eqs. G1-G2).

    Parameters
    ----------
    X_phys : np.ndarray of shape (N, d_phys)
        Physical Euclidean coordinates of all vertices.
    W_obs : np.ndarray of shape (N, N)
        Observed symmetric weighted adjacency matrix.
    subgraph_size : int, default=600
        Number of nodes sampled per subnetwork replica n (Table I, Appendix G.2).
    n_ensembles : int, default=40
        Number of independent Monte Carlo subgraphs K (Appendix G.2, Eq. G2).
    sampling_strategy : {'uniform', 'hub'}, default='uniform'
        - 'uniform': uniform random sampling across the entire vertex set (Eq. G1).
        - 'hub': sampling from top degree/strength hubs (used in heavy-tailed empirical networks).
    pool_size : int, optional
        Size of top hub candidate pool if sampling_strategy='hub' (e.g. 1200 for airports).
    soc_dim : int, default=2
        Dimensionality of the latent social space d_s.
    sigma_gauge : float, default=1.0
        Arbitrary gauge scale fixing the metric unit of social space.
    gamma_reg : float, default=0.25
        Tikhonov regularizer weight on (ln r - ln r_0)^2.
    adam_lr : float, default=0.04
        Adam learning rate.
    adam_steps : int, default=40
        Warm-up exploration budget.
    lbfgs_steps : int, default=40
        Quasi-Newton refinement budget.
    device : str or torch.device, optional
        Computation device.
    seed : int, optional
        Base random seed for reproducible sampling.
    verbose : bool, default=True
        Whether to print per-replica progress.

    Returns
    -------
    dict
        - 'lambda_mean': ensemble mean lambda_bar (Eq. G2).
        - 'lambda_std': ensemble sample standard deviation.
        - 'lambda_se': Monte Carlo standard error SE(lambda_bar) = std / sqrt(K).
        - 'r_mean': ensemble mean physical scale r_bar (Eq. G2).
        - 'r_std': ensemble sample standard deviation of r.
        - 'r_se': Monte Carlo standard error SE(r_bar) = std / sqrt(K).
        - 'ensemble_df': detailed DataFrame with per-replica records.
    """
    N = len(X_phys)
    if W_obs.shape != (N, N):
        raise ValueError(f"Shape mismatch: X has {N} rows, W has shape {W_obs.shape}")

    # Fallback to exact inference if graph size is within subgraph_size
    if N <= subgraph_size:
        fit = infer_parameters_exact(
            X_phys=X_phys,
            W_obs=W_obs,
            soc_dim=soc_dim,
            sigma_gauge=sigma_gauge,
            gamma_reg=gamma_reg,
            adam_lr=adam_lr,
            adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps,
            device=device,
            seed=seed,
        )
        return {
            "lambda_mean": fit["lambda"],
            "lambda_std": 0.0,
            "lambda_se": 0.0,
            "r_mean": fit["r"],
            "r_std": 0.0,
            "r_se": 0.0,
            "ensemble_df": pd.DataFrame([{
                "replica": 1,
                "n_nodes": N,
                "lambda": fit["lambda"],
                "r": fit["r"],
                "loss": fit["loss"],
            }]),
        }

    # Candidate node pool determination
    if sampling_strategy == "hub":
        strengths = W_obs.sum(axis=1)
        k_pool = pool_size if (pool_size is not None and pool_size < N) else min(1200, N)
        candidate_pool = np.argsort(-strengths)[:k_pool]
    elif sampling_strategy == "uniform":
        candidate_pool = np.arange(N)
    else:
        raise ValueError(f"Unknown sampling_strategy '{sampling_strategy}'. Choose 'uniform' or 'hub'.")

    base_rng = np.random.default_rng(seed)
    records = []

    for k in range(n_ensembles):
        t0 = time.time()
        rep_seed = None if seed is None else int(base_rng.integers(0, 100_000_000))
        rep_rng = np.random.default_rng(rep_seed)

        sampled_idx = rep_rng.choice(candidate_pool, size=subgraph_size, replace=False)
        X_sub = X_phys[sampled_idx]
        W_sub = W_obs[np.ix_(sampled_idx, sampled_idx)]

        sub_fit = infer_parameters_exact(
            X_phys=X_sub,
            W_obs=W_sub,
            soc_dim=soc_dim,
            sigma_gauge=sigma_gauge,
            gamma_reg=gamma_reg,
            adam_lr=adam_lr,
            adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps,
            compute_profile_ci=False,
            device=device,
            seed=rep_seed,
        )
        dt = time.time() - t0

        record = {
            "replica": k + 1,
            "n_nodes": subgraph_size,
            "lambda": sub_fit["lambda"],
            "r": sub_fit["r"],
            "loss": sub_fit["loss"],
            "runtime_s": dt,
        }
        records.append(record)

        if verbose:
            print(
                f"[Replica {k+1:2d}/{n_ensembles}] "
                f"lambda = {sub_fit['lambda']:.4f} | "
                f"r = {sub_fit['r']:7.2f} | "
                f"loss = {sub_fit['loss']:8.2f} | "
                f"time = {dt:.2f}s"
            )

    df_res = pd.DataFrame(records)
    lams = df_res["lambda"].values
    rs = df_res["r"].values
    K = len(lams)

    lam_mean = float(np.mean(lams))
    lam_std = float(np.std(lams, ddof=1)) if K > 1 else 0.0
    lam_se = float(lam_std / np.sqrt(K))

    r_mean = float(np.mean(rs))
    r_std = float(np.std(rs, ddof=1)) if K > 1 else 0.0
    r_se = float(r_std / np.sqrt(K))

    return {
        "lambda_mean": lam_mean,
        "lambda_std": lam_std,
        "lambda_se": lam_se,
        "r_mean": r_mean,
        "r_std": r_std,
        "r_se": r_se,
        "ensemble_df": df_res,
    }


if __name__ == "__main__":
    from spatiosocial_network_generator import generate_spatiosocial_network

    print("=" * 70)
    print("DEMO: Induced Subgraph Monte Carlo Ensembling (Appendix G.2)")
    print("=" * 70)

    true_lam = 0.45
    true_r = 0.22
    true_sigma = 0.15

    # Generate synthetic network with benchmark condition <s > = N
    net = generate_spatiosocial_network(
        n_nodes=400,
        lam=true_lam,
        r=true_r,
        sigma=true_sigma,
        n_gaussians=4,
        seed=123,
    )

    result = infer_parameters_subsampled(
        X_phys=net["X_phys"],
        W_obs=net["W"],
        subgraph_size=150,
        n_ensembles=5,
        sampling_strategy="uniform",
        seed=42,
        verbose=True,
    )

    print("\nEnsemble Aggregation (Eq. G2):")
    print(f"Ground Truth  : lambda = {true_lam:.3f} | r = {true_r:.3f}")
    print(f"Ensemble Mean : lambda_bar = {result['lambda_mean']:.3f} +/- {result['lambda_std']:.3f} (SE = {result['lambda_se']:.4f})")
    print(f"Ensemble Mean : r_bar      = {result['r_mean']:.3f} +/- {result['r_std']:.3f} (SE = {result['r_se']:.4f})")
