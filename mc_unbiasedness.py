from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np

from triangular_additive_model import simulate_option_A
from triangular_additive_model import fit_additive_triangular_wls
# or wherever these live


# -----------------------------
# Optional: pretty table output
# -----------------------------

def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


# ============================================================
# 1) A brute-force reference fit (explicit design matrix)
#    Useful for sanity-checking the optimizer implementation.
# ============================================================

def fit_additive_triangular_bruteforce_ols(
    panel,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brute-force weighted least squares by explicitly building the design matrix X
    for the model:
        Y_ij = a(T_i) + b(j), with constraint b(1)=0.

    Unknowns:
      - a(1..B)   : B params
      - b(2..B)   : B-1 params
      Total p = 2B-1

    Returns:
      a_hat: shape (B+1,), 1-based indexing (a_hat[0]=NaN)
      b_hat: shape (B+1,), 1-based indexing (b_hat[0]=NaN, b_hat[1]=0)
    """
    panel.validate()
    n, B = panel.n, panel.B

    if weights is None:
        w_user = np.ones(n, dtype=float)
    else:
        w_user = np.asarray(weights, dtype=float)
        if w_user.shape != (n,):
            raise ValueError("weights must have shape (n,)")
        if np.any(w_user < 0):
            raise ValueError("weights must be nonnegative")

    # Number of observed rows = sum_i T_i
    m = int(np.sum(panel.T))
    p = 2 * B - 1

    X = np.zeros((m, p), dtype=float)
    y = np.zeros(m, dtype=float)
    w_obs = np.zeros(m, dtype=float)

    row = 0
    for i in range(n):
        t = int(panel.T[i])
        Yi = panel.Y_list[i]
        wi = float(w_user[i])

        for j in range(1, t + 1):
            # a(t) column: index (t-1)
            X[row, t - 1] = 1.0

            # b(j) column for j>=2: block starts at B
            if j >= 2:
                X[row, B + (j - 2)] = 1.0

            y[row] = float(Yi[j - 1])
            w_obs[row] = wi
            row += 1

    # Weighted normal equations: (X^T W X) beta = (X^T W y)
    XtW = X.T * w_obs  # broadcast multiply each row by w_obs
    M = XtW @ X
    r = XtW @ y
    beta, *_ = np.linalg.lstsq(M, r, rcond=None)

    a_hat = np.full(B + 1, np.nan, dtype=float)
    b_hat = np.full(B + 1, np.nan, dtype=float)
    a_hat[1:] = beta[:B]
    b_hat[1] = 0.0
    b_hat[2:] = beta[B:]

    return a_hat, b_hat


def assert_fit_matches_bruteforce(
    panel,
    fit_fast_fn: Callable[..., Any],
    *,
    weights: Optional[np.ndarray] = None,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> None:
    """
    Compare your fast/exact solver to brute-force OLS on the same data.
    This is a very strong unit test for correctness.
    """
    a_ref, b_ref = fit_additive_triangular_bruteforce_ols(panel, weights=weights)
    fit = fit_fast_fn(panel, weights=weights, solver="lstsq", ridge=0.0, compute_objective=False)
    a_fast, b_fast = fit.a, fit.b

    # Compare only identifiable entries (non-NaN)
    mask_a = ~np.isnan(a_ref) & ~np.isnan(a_fast)
    mask_b = ~np.isnan(b_ref) & ~np.isnan(b_fast)

    if not np.allclose(a_ref[mask_a], a_fast[mask_a], atol=atol, rtol=rtol):
        raise AssertionError("Fast solver a(t) does not match brute-force OLS.")
    if not np.allclose(b_ref[mask_b], b_fast[mask_b], atol=atol, rtol=rtol):
        raise AssertionError("Fast solver b(j) does not match brute-force OLS.")

    # Ensure constraint is met
    if not (abs(b_fast[1]) < 1e-12):
        raise AssertionError(f"Constraint violated: b_fast[1]={b_fast[1]}")

    print("✅ Sanity check passed: fast solver matches brute-force OLS.")


# ============================================================
# 2) Monte Carlo unbiasedness harness for Option A
# ============================================================

@dataclass(frozen=True)
class MCConfig:
    R: int = 2000          # number of replications
    n: int = 20000         # users per replication
    B: int = 8             # max exposure index
    seed: int = 12345
    correlate_u_A: bool = True  # use corr_u_A > 0 or 0
    show_progress: bool = True
    sanity_check_first_rep: bool = True  # compares solver to brute-force on a small subsample
    sanity_check_n: int = 2000           # smaller n for brute-force check


def mc_unbiasedness_option_A(
    simulate_option_A_fn: Callable[..., Tuple[Any, np.ndarray, np.ndarray]],
    fit_fn: Callable[..., Any],
    *,
    cfg: MCConfig = MCConfig(),
    # Pass through simulation params (override defaults if you like)
    lam0: float = 2.0,
    gamma: float = 0.9,
    tau_u: float = 0.6,
    corr_u_A: float = 0.35,
    rho: float = 0.7,
    sigma0: float = 1.2,
    sigma_decay: float = 0.05,
    # Fitter params
    solver: str = "lstsq",
    ridge: float = 0.0,
) -> Any:
    """
    Runs R Monte Carlo replications under Option A and estimates bias of b_hat(j).

    Reports for each j:
      - true b(j)
      - Monte Carlo mean of b_hat(j)
      - estimated bias = mean(b_hat(j)) - b_true(j)
      - Monte Carlo SE of the mean
      - z-score = bias / MC_SE
      - 95% CI for bias

    Interpretation:
      If estimator is unbiased, z-scores should look like N(0,1) noise.
      With big R, the CI for bias should be very tight around 0.

    Returns:
      - pandas DataFrame if pandas is installed, else a dict of numpy arrays.
    """
    R, n, B = cfg.R, cfg.n, cfg.B

    # Spawn independent RNG streams (best practice)
    ss = np.random.SeedSequence(cfg.seed)
    rep_seeds = ss.spawn(R)

    # Storage (NaNs are helpful if something is unidentifiable in a replication)
    b_hat_mat = np.full((R, B + 1), np.nan, dtype=float)

    b_true_global = None
    n_fail = 0

    for r in range(R):
        if cfg.show_progress and (r % max(1, R // 20) == 0):
            print(f"[MC] replication {r+1}/{R}")

        # Each replication gets an independent SeedSequence
        rep_seed = rep_seeds[r]

        # Choose correlation
        corr = corr_u_A if cfg.correlate_u_A else 0.0

        panel, _a_true, b_true = simulate_option_A_fn(
            n=n, B=B, seed=rep_seed,
            lam0=lam0, gamma=gamma,
            corr_u_A=corr, tau_u=tau_u,
            rho=rho, sigma0=sigma0, sigma_decay=sigma_decay,
        )

        if b_true_global is None:
            b_true_global = b_true.copy()

            # Optional: brute-force solver sanity check on smaller n
            if cfg.sanity_check_first_rep:
                panel_small, *_ = simulate_option_A_fn(
                    n=min(cfg.sanity_check_n, n), B=B, seed=rep_seed,
                    lam0=lam0, gamma=gamma,
                    corr_u_A=corr, tau_u=tau_u,
                    rho=rho, sigma0=sigma0, sigma_decay=sigma_decay,
                )
                assert_fit_matches_bruteforce(panel_small, fit_fn)

        try:
            fit = fit_fn(panel, weights=None, solver=solver, ridge=ridge, compute_objective=False)
            b_hat = fit.b
            # Hard check on the constraint
            if not (abs(b_hat[1]) < 1e-10):
                raise RuntimeError(f"Constraint violated: b_hat[1]={b_hat[1]}")
            b_hat_mat[r, :] = b_hat
        except Exception as e:
            n_fail += 1
            if cfg.show_progress:
                print(f"[MC] replication {r+1} FAILED: {repr(e)}")
            continue

    if b_true_global is None:
        raise RuntimeError("MC run produced no successful replications.")

    if n_fail > 0:
        print(f"[MC] WARNING: {n_fail}/{R} replications failed (see logs).")

    # Compute Monte Carlo summaries (nan-safe)
    mean_hat = np.nanmean(b_hat_mat, axis=0)
    sd_hat = np.nanstd(b_hat_mat, axis=0, ddof=1)
    n_used = np.sum(~np.isnan(b_hat_mat), axis=0).astype(int)

    mc_se = sd_hat / np.sqrt(np.maximum(n_used, 1))
    bias = mean_hat - b_true_global

    # z-score for bias estimate (skip j where mc_se=0)
    z = np.full(B + 1, np.nan, dtype=float)
    ok = (mc_se > 0) & (~np.isnan(bias))
    z[ok] = bias[ok] / mc_se[ok]

    # 95% CI for bias
    ci_lo = bias - 1.96 * mc_se
    ci_hi = bias + 1.96 * mc_se

    # A compact overall diagnostic: max |z| over j>=2
    max_abs_z = np.nanmax(np.abs(z[2:]))

    print(f"[MC] max |z| over j>=2: {max_abs_z:.3f}")
    print("[MC] If unbiased + enough R, these z-scores should look like noise around 0.\n")

    pd = _try_import_pandas()
    if pd is not None:
        df = pd.DataFrame({
            "j": np.arange(1, B + 1),
            "b_true": b_true_global[1:],
            "mean_b_hat": mean_hat[1:],
            "bias": bias[1:],
            "mc_se_mean": mc_se[1:],
            "z": z[1:],
            "ci95_lo_bias": ci_lo[1:],
            "ci95_hi_bias": ci_hi[1:],
            "n_used": n_used[1:],
        })
        return df
    else:
        return {
            "j": np.arange(1, B + 1),
            "b_true": b_true_global,
            "mean_b_hat": mean_hat,
            "bias": bias,
            "mc_se_mean": mc_se,
            "z": z,
            "ci95_lo_bias": ci_lo,
            "ci95_hi_bias": ci_hi,
            "n_used": n_used,
            "b_hat_mat": b_hat_mat,  # keep if you want (can be big)
        }
    

def main():
    cfg = MCConfig(
        R=2000,
        n=20000,
        B=8,
        seed=20260304,
        correlate_u_A=True,
        show_progress=True,
        sanity_check_first_rep=True,
        sanity_check_n=2000,
    )

    df = mc_unbiasedness_option_A(
        simulate_option_A_fn=simulate_option_A,
        fit_fn=fit_additive_triangular_wls,
        cfg=cfg,
        sigma_decay=0.05,
    )

    # If pandas exists, df is a DataFrame; otherwise it's a dict.
    try:
        print(df.to_string(index=False))  # pandas
    except AttributeError:
        # dict fallback
        for j, bias, z in zip(df["j"][1:], df["bias"][1:], df["z"][1:]):
            print(f"j={j:2d}  bias={bias: .6g}  z={z: .3f}")

if __name__ == "__main__":
    main()