#!/usr/bin/env python3
"""Grid experiment runner for the triangular additive model.

Requested grid:
  - B in {5, 30}
  - n in {1000, 10000}
  - SNR in {0.3, 0.2, 0.1}, where
      SNR := (max_k b_true(k) - min_k b_true(k)) / sd(u_i + eps_ij)

For now we set within-user correlation to 0 by using rho=0.
To make the *within-user* dependence truly 0, we also set tau_u=0 by default;
then sd(u_i + eps_ij) = sd(eps_ij) = sigma0.

Methods compared:
  - "ours": exact WLS additive fit with constraint b(1)=0
  - "naive": column means (no imputation) shifted to b(1)=0

Metrics (per j and aggregated):
  - Monte Carlo sd of estimator
  - Monte Carlo bias
  - Wald 95% coverage using estimated sigma^2 (ours) and plug-in sigma^2 (naive)

Note: Even if an estimator is unbiased in theory, we still need repeated sampling
(Monte Carlo) to estimate its sd and coverage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from triangular_additive_model import (
    OptionAParams,
    default_truth_curves,
    fit_additive_triangular_wls,
    naive_b_curve,
    simulate_option_A,
)


Z_975 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


@dataclass(frozen=True)
class GridConfig:
    R: int
    seed: int
    B_list: Tuple[int, ...]
    n_list: Tuple[int, ...]
    snr_list: Tuple[float, ...]


def _snr_amplitude(b_true: np.ndarray) -> float:
    vals = b_true[1:]
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals) - np.min(vals))


def make_params_for_snr(
    *,
    B: int,
    snr: float,
    lam0: float = 2.0,
    gamma: float = 0.9,
    corr_u_A: float = 0.0,
    tau_u: float = 0.0,
    rho: float = 0.0,
    sigma_decay: float = 0.0,
) -> OptionAParams:
    """Create OptionAParams with sigma0 chosen to match the target SNR."""
    if snr <= 0:
        raise ValueError("snr must be > 0")

    _a_true, b_true = default_truth_curves(B)
    amp = _snr_amplitude(b_true)

    # target sd(u + eps) = amp / snr.
    target_sd = amp / snr

    # With rho=0 and sigma_decay=0, eps_ij ~ N(0, sigma0^2).
    # If tau_u>0, sd(u + eps) = sqrt(tau_u^2 + sigma0^2).
    var_eps = target_sd**2 - tau_u**2
    if var_eps <= 0:
        raise ValueError(
            f"Infeasible: target_sd^2={target_sd**2:.6g} <= tau_u^2={tau_u**2:.6g}. "
            "Decrease tau_u or increase noise (lower SNR)."
        )
    sigma0 = float(np.sqrt(var_eps))

    return OptionAParams(
        B=B,
        lam0=lam0,
        gamma=gamma,
        corr_u_A=corr_u_A,
        tau_u=tau_u,
        rho=rho,
        sigma0=sigma0,
        sigma_decay=sigma_decay,
    )


def _build_normal_matrix_b1_zero(panel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    """Rebuild X'X and X'y for the additive model with b(1)=0 (unweighted)."""
    panel.validate()
    n, B = panel.n, panel.B

    # counts by T
    N_t = np.bincount(panel.T, minlength=B + 1).astype(float)
    # counts by support at exposure j: N_ge[j] = # {i: T_i >= j}
    N_ge = np.cumsum(N_t[::-1])[::-1]

    active_t = np.array([t for t in range(1, B + 1) if N_t[t] > 0], dtype=int)
    active_j = np.array([j for j in range(2, B + 1) if N_ge[j] > 0], dtype=int)

    p = active_t.size + active_j.size
    idx_a = {t: k for k, t in enumerate(active_t)}
    idx_b = {j: active_t.size + k for k, j in enumerate(active_j)}

    M = np.zeros((p, p), dtype=float)
    r = np.zeros(p, dtype=float)

    # Y aggregates
    Ysum_t = np.zeros(B + 1, dtype=float)
    Ysum_j = np.zeros(B + 1, dtype=float)
    for i in range(n):
        t = int(panel.T[i])
        Yi = panel.Y_list[i]
        Ysum_t[t] += float(np.sum(Yi))
        Ysum_j[1 : t + 1] += Yi

    # a-block
    for t in active_t:
        ia = idx_a[t]
        M[ia, ia] = t * N_t[t]
        r[ia] = Ysum_t[t]

    # b-block
    for j in active_j:
        ib = idx_b[j]
        M[ib, ib] = N_ge[j]
        r[ib] = Ysum_j[j]

    # cross terms
    for t in active_t:
        ia = idx_a[t]
        mass = N_t[t]
        if mass <= 0:
            continue
        for j in active_j:
            if j <= t:
                ib = idx_b[j]
                M[ia, ib] = mass
                M[ib, ia] = mass

    return M, r, active_t, active_j, idx_b


def se_b_ours_homoskedastic(panel, fit, sigma2: float) -> np.ndarray:
    """Compute homoskedastic SE for b_hat under correct iid noise."""
    M, _r, active_t, active_j, idx_b = _build_normal_matrix_b1_zero(panel)

    # Use pseudo-inverse to be robust (though with support it should be full rank).
    Minv = np.linalg.pinv(M)

    B = panel.B
    se = np.full(B + 1, np.nan, dtype=float)
    se[1] = 0.0

    for j in active_j:
        ib = idx_b[j]
        se[j] = float(np.sqrt(sigma2 * Minv[ib, ib]))

    return se


def sigma2_hat_from_fit(panel, fit) -> Tuple[float, int]:
    """Estimate sigma^2 from residuals; returns (sigma2_hat, df)."""
    panel.validate()
    B = panel.B

    # number of obs
    n_obs = int(np.sum(panel.T))

    # degrees of freedom: n_obs - (num active a's + num active b's)
    active_t = np.asarray(fit.diagnostics.get("active_t", []), dtype=int)
    active_j = np.asarray(fit.diagnostics.get("active_j", []), dtype=int)
    p = int(active_t.size + active_j.size)

    df = max(1, n_obs - p)

    sse = 0.0
    a = fit.a
    b = fit.b
    for i in range(panel.n):
        t = int(panel.T[i])
        Yi = panel.Y_list[i]
        pred = a[t] + b[1 : t + 1]
        resid = Yi - pred
        sse += float(np.sum(resid * resid))

    return float(sse / df), df


def se_b_naive(panel, sigma2: float) -> np.ndarray:
    """Approx SE for naive b(j) = mean(Y_ij | T>=j) - mean(Y_i1).

    Under tau_u=0 and rho=0, eps are independent across i and j,
    so Var(mean_j - mean_1) = sigma2*(1/N_j + 1/n).

    (We use the observed N_j in the sample.)
    """
    panel.validate()
    n, B = panel.n, panel.B

    N_t = np.bincount(panel.T, minlength=B + 1).astype(int)
    N_ge = np.cumsum(N_t[::-1])[::-1].astype(int)

    se = np.full(B + 1, np.nan, dtype=float)
    se[1] = 0.0

    for j in range(2, B + 1):
        Nj = int(N_ge[j])
        if Nj <= 0:
            continue
        se[j] = float(np.sqrt(sigma2 * (1.0 / Nj + 1.0 / n)))

    return se


def run_grid(cfg: GridConfig) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    ss = np.random.SeedSequence(cfg.seed)

    for B in cfg.B_list:
        a_true, b_true = default_truth_curves(B)
        amp = _snr_amplitude(b_true)

        for snr in cfg.snr_list:
            params = make_params_for_snr(
                B=B,
                snr=snr,
                lam0=2.0,
                gamma=0.9,
                corr_u_A=0.0,
                tau_u=0.0,
                rho=0.0,
                sigma_decay=0.0,
            )

            for n in cfg.n_list:
                rep_seeds = ss.spawn(cfg.R)

                b_hat_mat = np.full((cfg.R, B + 1), np.nan, dtype=float)
                b_naive_mat = np.full((cfg.R, B + 1), np.nan, dtype=float)
                cover_hat = np.full((cfg.R, B + 1), np.nan, dtype=float)
                cover_naive = np.full((cfg.R, B + 1), np.nan, dtype=float)

                for r in range(cfg.R):
                    panel, _a, _b, _lat = simulate_option_A(n=n, params=params, seed=rep_seeds[r], return_latents=False)

                    fit = fit_additive_triangular_wls(panel, solver="lstsq", ridge=0.0, compute_objective=False)
                    b_hat = fit.b
                    b_naive = naive_b_curve(panel)

                    sigma2_hat, _df = sigma2_hat_from_fit(panel, fit)

                    se_hat = se_b_ours_homoskedastic(panel, fit, sigma2=sigma2_hat)
                    se_nv = se_b_naive(panel, sigma2=sigma2_hat)

                    lo_hat = b_hat - Z_975 * se_hat
                    hi_hat = b_hat + Z_975 * se_hat
                    lo_nv = b_naive - Z_975 * se_nv
                    hi_nv = b_naive + Z_975 * se_nv

                    b_hat_mat[r, :] = b_hat
                    b_naive_mat[r, :] = b_naive

                    # coverage (skip NaNs)
                    cover_hat[r, :] = (b_true >= lo_hat) & (b_true <= hi_hat)
                    cover_naive[r, :] = (b_true >= lo_nv) & (b_true <= hi_nv)

                # aggregate metrics over reps
                bias_hat = np.nanmean(b_hat_mat - b_true[None, :], axis=0)
                bias_nv = np.nanmean(b_naive_mat - b_true[None, :], axis=0)
                sd_hat = np.nanstd(b_hat_mat, axis=0, ddof=1)
                sd_nv = np.nanstd(b_naive_mat, axis=0, ddof=1)
                cov_hat = np.nanmean(cover_hat, axis=0)
                cov_nv = np.nanmean(cover_naive, axis=0)

                # summarize across j=2..B (exclude ident constraint j=1)
                js = np.arange(2, B + 1)
                row = {
                    "B": B,
                    "n": n,
                    "snr": snr,
                    "amp": amp,
                    "sigma0": params.sigma0,
                    "R": cfg.R,
                    "ours_mean_bias": float(np.nanmean(bias_hat[js])),
                    "ours_mean_sd": float(np.nanmean(sd_hat[js])),
                    "ours_mean_cov": float(np.nanmean(cov_hat[js])),
                    "naive_mean_bias": float(np.nanmean(bias_nv[js])),
                    "naive_mean_sd": float(np.nanmean(sd_nv[js])),
                    "naive_mean_cov": float(np.nanmean(cov_nv[js])),
                }
                rows.append(row)

                # Also print a compact per-config line for quick feedback
                print(
                    f"B={B:2d} n={n:6d} snr={snr:.1f} sigma0={params.sigma0:.3f} | "
                    f"bias(ours)={row['ours_mean_bias']:+.4f} sd(ours)={row['ours_mean_sd']:.4f} cov(ours)={row['ours_mean_cov']:.3f} | "
                    f"bias(naive)={row['naive_mean_bias']:+.4f} sd(naive)={row['naive_mean_sd']:.4f} cov(naive)={row['naive_mean_cov']:.3f}"
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grid experiments for triangular additive model")
    parser.add_argument("--R", type=int, default=300, help="Monte Carlo replications per config")
    parser.add_argument("--seed", type=int, default=20260305, help="base RNG seed")
    parser.add_argument("--B", type=str, default="5,30", help="comma-separated B values")
    parser.add_argument("--n", type=str, default="1000,10000", help="comma-separated n values")
    parser.add_argument("--snr", type=str, default="0.3,0.2,0.1", help="comma-separated SNR targets")
    parser.add_argument("--no_pandas", action="store_true", help="do not use pandas even if installed")
    args = parser.parse_args()

    B_list = tuple(int(x) for x in args.B.split(",") if x.strip())
    n_list = tuple(int(x) for x in args.n.split(",") if x.strip())
    snr_list = tuple(float(x) for x in args.snr.split(",") if x.strip())

    cfg = GridConfig(R=args.R, seed=args.seed, B_list=B_list, n_list=n_list, snr_list=snr_list)
    rows = run_grid(cfg)

    if args.no_pandas:
        return

    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        # stable ordering
        df = df.sort_values(["B", "n", "snr"], ascending=[True, True, False])
        print("\nSummary (averaged over j=2..B):")
        print(df.to_string(index=False))
    except Exception:
        # pandas not installed; nothing else to do.
        pass


if __name__ == "__main__":
    main()
