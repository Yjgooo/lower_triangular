#!/usr/bin/env python3
"""
paper_experiment_grid.py

Monte Carlo experiment grid for triangular novelty-curve estimation.

Methods:
  - Our method: additive triangular LS fit with b(1)=0, bootstrap CIs by resampling users.
  - Naive method: column means without filling missing cells, with naive i.i.d.-assumption CI.

Grid:
  B in {5, 30}
  n in {1000, 10000}
  SNR in {0.3, 0.2, 0.1}
  ICC fixed at 0.20
  rho in {0.3, 0.6, 0.8}

Metrics:
  SD, Bias, MSE, Coverage (per j), plus support diagnostics N_j.

Notes:
  - This file prioritizes clarity and correctness. It is not aggressively optimized.
  - Nested MC + bootstrap can be computationally expensive; tune R_mc and R_boot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np

# CHANGE THIS import if your estimator module has a different name.
from triangular_additive_model import TriangularPanel, fit_additive_triangular_wls


# ============================================================
# 1) Configuration dataclasses
# ============================================================

@dataclass(frozen=True)
class DependenceSpec:
    """Within-user dependence settings."""
    icc: float = 0.20
    rho: float = 0.6


@dataclass(frozen=True)
class Scenario:
    """One scenario in the experiment grid."""
    B: int
    n: int
    snr: float
    dep: DependenceSpec


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Controls Monte Carlo / bootstrap repetition counts and other defaults.
    """
    R_mc: int = 200          # Monte Carlo replications per scenario
    R_boot: int = 200        # Bootstrap draws per replication (our method)
    alpha: float = 0.05      # CI level: 1-alpha
    seed: int = 12345

    # Length model: truncated geometric is chosen for good tail support for B=30.
    # np.random.Generator.geometric(p) returns values in {1,2,...} with mean 1/p.
    # Default p = 2/(B+1) -> mean about (B+1)/2 (good support up to B).
    length_p_override: Optional[float] = None

    # For “naive i.i.d.” CI:
    z_crit: float = 1.96     # approximate N(0,1) 97.5% quantile for 95% CI

    # For handling extremely low support:
    # If N_j < min_support, mark estimates/CI as NaN for that j in that replication.
    min_support: int = 5

    # Solver settings for our method
    solver: str = "lstsq"
    ridge: float = 0.0


# ============================================================
# 2) Truth curves
# ============================================================

def make_truth_curves(B: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (a_true, b_true) as 1-based arrays of shape (B+1,) with:
      - a_true[t] for t=1..B
      - b_true[1]=0 and b_true[j] for j=2..B
    """
    tgrid = np.arange(1, B + 1)
    a = 0.25 * np.log1p(tgrid)

    jgrid = np.arange(1, B + 1)
    # saturating negative curve, anchored at b(1)=0
    b = -0.6 * (1.0 - np.exp(-(jgrid - 1) / 6.0))
    b[0] = 0.0

    a1 = np.full(B + 1, np.nan, dtype=float)
    b1 = np.full(B + 1, np.nan, dtype=float)
    a1[1:] = a
    b1[1:] = b
    return a1, b1


# ============================================================
# 3) Length model (no individual covariates)
# ============================================================

def sample_lengths_truncated_geometric(
    n: int,
    B: int,
    rng: np.random.Generator,
    p: Optional[float] = None,
) -> np.ndarray:
    """
    Sample T_i ~ min(Geom(p), B), where np.Generator.geometric(p) has support {1,2,...}.

    Default p = 2/(B+1) gives E[T] ~ (B+1)/2 before truncation.
    This gives meaningful tail support for B=30 even at n=1000.
    """
    if p is None:
        p = 2.0 / (B + 1.0)
    if not (0.0 < p <= 1.0):
        raise ValueError(f"geometric p must be in (0,1], got {p}")

    T = rng.geometric(p, size=n).astype(int)
    T = np.clip(T, 1, B)
    return T


def exposure_support_counts(T: np.ndarray, B: int) -> np.ndarray:
    """
    N_j = number of users with T_i >= j, for j=1..B (1-based array).
    """
    counts_by_t = np.bincount(T, minlength=B + 1)
    # suffix sum: N_j = sum_{t>=j} count(t)
    Nj = np.cumsum(counts_by_t[::-1])[::-1]
    return Nj.astype(int)


# ============================================================
# 4) Noise calibration to satisfy SNR and ICC, and AR(1) simulation
# ============================================================

@dataclass(frozen=True)
class NoiseParams:
    sigma_tot: float     # target sd(u + epsilon)
    tau_u: float         # sd of u
    sigma_eps: float     # marginal sd of stationary epsilon
    sigma_eta: float     # innovation sd in AR(1)


def calibrate_noise_from_snr_icc_rho(
    b_true: np.ndarray,
    snr: float,
    icc: float,
    rho: float,
) -> NoiseParams:
    """
    SNR = (max b - min b) / sd(u + eps)

    Enforce:
      Var(u) = ICC * sigma_tot^2
      Var(eps) = (1-ICC) * sigma_tot^2
    And stationary AR(1):
      Var(eps) = sigma_eta^2 / (1-rho^2)  => sigma_eta = sigma_eps * sqrt(1-rho^2)
    """
    if snr <= 0:
        raise ValueError("snr must be > 0")
    if not (0.0 < icc < 1.0):
        raise ValueError("icc must be in (0,1)")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be in (-1,1)")

    # range of b over j=1..B
    b_vals = b_true[1:]
    delta_b = float(np.nanmax(b_vals) - np.nanmin(b_vals))
    if delta_b <= 0:
        raise ValueError("b_true must have positive range for SNR calibration")

    sigma_tot = delta_b / snr
    tau_u = sigma_tot * math.sqrt(icc)
    sigma_eps = sigma_tot * math.sqrt(1.0 - icc)
    sigma_eta = sigma_eps * math.sqrt(1.0 - rho * rho)

    return NoiseParams(sigma_tot=sigma_tot, tau_u=tau_u, sigma_eps=sigma_eps, sigma_eta=sigma_eta)


def simulate_panel_one_replication(
    *,
    n: int,
    B: int,
    a_true: np.ndarray,
    b_true: np.ndarray,
    snr: float,
    dep: DependenceSpec,
    rng: np.random.Generator,
    length_p: Optional[float] = None,
) -> Tuple[TriangularPanel, NoiseParams, np.ndarray]:
    """
    Simulate one triangular panel replication with:
      Y_ij = a_true(T_i) + b_true(j) + u_i + eps_ij
      u_i ~ N(0, tau_u^2)
      eps_ij stationary AR(1) with parameter rho and marginal var sigma_eps^2
    Returns:
      panel, noise_params, Nj (support counts)
    """
    a_true = np.asarray(a_true, dtype=float)
    b_true = np.asarray(b_true, dtype=float)

    # lengths
    T = sample_lengths_truncated_geometric(n=n, B=B, rng=rng, p=length_p)
    Nj = exposure_support_counts(T, B)

    # noise calibration
    noise = calibrate_noise_from_snr_icc_rho(b_true=b_true, snr=snr, icc=dep.icc, rho=dep.rho)

    # random intercepts
    u = rng.normal(0.0, noise.tau_u, size=n)

    # generate Y_list
    Y_list: List[np.ndarray] = []
    for i in range(n):
        t = int(T[i])

        # stationary initialization of eps
        eps_prev = rng.normal(0.0, noise.sigma_eps)

        Yi = np.empty(t, dtype=float)
        for jj in range(1, t + 1):
            eta = rng.normal(0.0, noise.sigma_eta)
            eps = dep.rho * eps_prev + eta
            eps_prev = eps
            Yi[jj - 1] = a_true[t] + b_true[jj] + u[i] + eps

        Y_list.append(Yi)

    panel = TriangularPanel(Y_list=Y_list, T=T, B=B).validate()
    return panel, noise, Nj


# ============================================================
# 5) Naive estimator + naive i.i.d. CI
# ============================================================

@dataclass(frozen=True)
class NaiveStats:
    """
    Column-wise summary stats among observed cells (users with T>=j).
    All arrays are 1-based of shape (B+1,), index 0 unused.
    """
    mean_y: np.ndarray
    var_y: np.ndarray
    Nj: np.ndarray


def compute_naive_column_stats(panel: TriangularPanel) -> NaiveStats:
    """
    Compute, for each exposure j:
      mean_y[j] = mean of Y_ij among users with T_i >= j
      var_y[j]  = sample variance (ddof=1) among same users
      Nj[j]     = count of users with T_i >= j

    Uses one pass over users accumulating sums and sums of squares.
    """
    panel.validate()
    n, B = panel.n, panel.B

    sum_y = np.zeros(B + 1, dtype=float)
    sum_y2 = np.zeros(B + 1, dtype=float)
    Nj = np.zeros(B + 1, dtype=int)

    for i in range(n):
        t = int(panel.T[i])
        Yi = panel.Y_list[i]
        # add to all j<=t
        sum_y[1 : t + 1] += Yi
        sum_y2[1 : t + 1] += Yi * Yi
        Nj[1 : t + 1] += 1

    mean_y = np.full(B + 1, np.nan, dtype=float)
    var_y = np.full(B + 1, np.nan, dtype=float)

    # mean where Nj>0
    ok_mean = Nj[1:] > 0
    mean_y[1:][ok_mean] = sum_y[1:][ok_mean] / Nj[1:][ok_mean]

    # sample variance where Nj>=2: (sum_y2 - N*mean^2)/(N-1)
    ok_var = Nj[1:] >= 2
    numer = sum_y2[1:][ok_var] - Nj[1:][ok_var] * (mean_y[1:][ok_var] ** 2)
    # numerical guard
    numer = np.maximum(numer, 0.0)
    var_y[1:][ok_var] = numer / (Nj[1:][ok_var] - 1)

    return NaiveStats(mean_y=mean_y, var_y=var_y, Nj=Nj)


def naive_estimate_and_iid_ci(
    panel: TriangularPanel,
    *,
    alpha: float,
    z_crit: float,
    min_support: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Naive estimator:
      b_naive(j) = mean_y[j] - mean_y[1]  => b_naive(1)=0

    Naive i.i.d.-assumption CI:
      SE(b_naive(j)) = sqrt( var_y[j]/Nj[j] + var_y[1]/Nj[1] )
      CI = b_naive(j) ± z_crit * SE

    Important: This CI ignores within-user dependence and overlap in samples.
    """
    stats = compute_naive_column_stats(panel)
    B = panel.B

    b_hat = np.full(B + 1, np.nan, dtype=float)
    lo = np.full(B + 1, np.nan, dtype=float)
    hi = np.full(B + 1, np.nan, dtype=float)

    # Require basic support
    if stats.Nj[1] < max(min_support, 2):
        # No way to define a variance for j=1; return all-NaN
        return b_hat, lo, hi, stats.Nj

    # define b_hat
    b_hat[1:] = stats.mean_y[1:] - stats.mean_y[1]
    b_hat[1] = 0.0

    # standard errors
    se = np.full(B + 1, np.nan, dtype=float)

    var1 = stats.var_y[1]
    n1 = stats.Nj[1]
    for j in range(1, B + 1):
        nj = stats.Nj[j]
        if nj < max(min_support, 2):
            continue
        varj = stats.var_y[j]
        if not (np.isfinite(varj) and np.isfinite(var1)):
            continue
        se[j] = math.sqrt(varj / nj + var1 / n1)

    # CI (normal approx)
    for j in range(1, B + 1):
        if not np.isfinite(se[j]):
            continue
        lo[j] = b_hat[j] - z_crit * se[j]
        hi[j] = b_hat[j] + z_crit * se[j]

    return b_hat, lo, hi, stats.Nj


# ============================================================
# 6) Our method + standard cluster bootstrap percentile CI
# ============================================================

def fit_ours(panel: TriangularPanel, weights: Optional[np.ndarray], cfg: ExperimentConfig) -> np.ndarray:
    """
    Fit our method and return b_hat (1-based array).
    """
    fit = fit_additive_triangular_wls(
        panel,
        weights=weights,
        solver=cfg.solver,
        ridge=cfg.ridge,
        compute_objective=False,
    )
    return fit.b


def bootstrap_ci_ours(
    panel: TriangularPanel,
    *,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard cluster bootstrap:
      - resample users with replacement (multinomial counts),
      - refit our method,
      - percentile CI per j.

    Returns:
      lo, hi as 1-based arrays shape (B+1,)
    """
    panel.validate()
    n, B = panel.n, panel.B
    Rb = cfg.R_boot
    alpha = cfg.alpha

    b_boot = np.full((Rb, B + 1), np.nan, dtype=float)

    for r in range(Rb):
        # resample users with replacement
        idx = rng.integers(0, n, size=n)
        counts = np.bincount(idx, minlength=n).astype(float)

        b_hat_star = fit_ours(panel, weights=counts, cfg=cfg)
        b_boot[r, :] = b_hat_star

    lo = np.full(B + 1, np.nan, dtype=float)
    hi = np.full(B + 1, np.nan, dtype=float)

    # Ignore index 0 by design; compute quantiles for 1..B
    # (index 1 is always 0, but we keep it for completeness)
    lo[1:] = np.nanquantile(b_boot[:, 1:], alpha / 2.0, axis=0)
    hi[1:] = np.nanquantile(b_boot[:, 1:], 1.0 - alpha / 2.0, axis=0)

    return lo, hi


# ============================================================
# 7) Scenario evaluation (MC loop)
# ============================================================

def evaluate_scenario(
    scen: Scenario,
    cfg: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """
    Run R_mc Monte Carlo replications for one scenario.
    Returns a list of row dicts (tidy format) for both methods and each j.
    """
    B, n, snr = scen.B, scen.n, scen.snr
    icc, rho = scen.dep.icc, scen.dep.rho

    a_true, b_true = make_truth_curves(B)

    # scenario-specific seed sequence (stable regardless of run ordering)
    scen_seed = np.random.SeedSequence([
        cfg.seed,
        int(B),
        int(n),
        int(round(1000 * snr)),
        int(round(1000 * icc)),
        int(round(1000 * rho)),
    ])
    rep_seeds = scen_seed.spawn(cfg.R_mc)

    # storage (MC x (B+1))
    b_ours = np.full((cfg.R_mc, B + 1), np.nan, dtype=float)
    lo_ours = np.full((cfg.R_mc, B + 1), np.nan, dtype=float)
    hi_ours = np.full((cfg.R_mc, B + 1), np.nan, dtype=float)

    b_naive = np.full((cfg.R_mc, B + 1), np.nan, dtype=float)
    lo_naive = np.full((cfg.R_mc, B + 1), np.nan, dtype=float)
    hi_naive = np.full((cfg.R_mc, B + 1), np.nan, dtype=float)

    Nj_mat = np.full((cfg.R_mc, B + 1), np.nan, dtype=float)

    # fixed length parameter rule unless overridden
    length_p = cfg.length_p_override
    if length_p is None:
        length_p = 2.0 / (B + 1.0)

    for r, ss in enumerate(rep_seeds):
        # split seed: one for simulation, one for bootstrap
        ss_sim, ss_boot = ss.spawn(2)
        rng_sim = np.random.default_rng(ss_sim)
        rng_boot = np.random.default_rng(ss_boot)

        panel, _noise, Nj = simulate_panel_one_replication(
            n=n,
            B=B,
            a_true=a_true,
            b_true=b_true,
            snr=snr,
            dep=scen.dep,
            rng=rng_sim,
            length_p=length_p,
        )
        Nj_mat[r, :] = Nj

        # Our method point estimate
        b_hat = fit_ours(panel, weights=None, cfg=cfg)
        b_ours[r, :] = b_hat

        # Our method bootstrap CI
        lo_b, hi_b = bootstrap_ci_ours(panel, cfg=cfg, rng=rng_boot)
        lo_ours[r, :] = lo_b
        hi_ours[r, :] = hi_b

        # Naive method + naive i.i.d. CI
        bN, loN, hiN, _NjN = naive_estimate_and_iid_ci(
            panel,
            alpha=cfg.alpha,
            z_crit=cfg.z_crit,
            min_support=cfg.min_support,
        )
        b_naive[r, :] = bN
        lo_naive[r, :] = loN
        hi_naive[r, :] = hiN

    # Aggregate metrics per j and method
    rows: List[Dict[str, Any]] = []
    for method in ("ours", "naive"):
        if method == "ours":
            bh = b_ours
            lo = lo_ours
            hi = hi_ours
        else:
            bh = b_naive
            lo = lo_naive
            hi = hi_naive

        for j in range(1, B + 1):
            true_j = b_true[j]
            if not np.isfinite(true_j):
                continue

            # keep only replications where estimate is finite and CI is defined
            ok = np.isfinite(bh[:, j]) & np.isfinite(lo[:, j]) & np.isfinite(hi[:, j])
            n_used = int(np.sum(ok))
            if n_used == 0:
                rows.append({
                    "B": B, "n": n, "snr": snr, "icc": icc, "rho": rho,
                    "method": method, "j": j,
                    "sd": np.nan, "bias": np.nan, "mse": np.nan, "coverage": np.nan,
                    "n_rep_used": 0,
                    "mean_Nj": float(np.nanmean(Nj_mat[:, j])),
                    "min_Nj": float(np.nanmin(Nj_mat[:, j])),
                })
                continue

            est = bh[ok, j]
            err = est - true_j

            sd = float(np.std(est, ddof=1)) if n_used >= 2 else np.nan
            bias = float(np.mean(err))
            mse = float(np.mean(err * err))

            cov = float(np.mean((lo[ok, j] <= true_j) & (true_j <= hi[ok, j])))

            rows.append({
                "B": B, "n": n, "snr": snr, "icc": icc, "rho": rho,
                "method": method, "j": j,
                "sd": sd,
                "bias": bias,
                "mse": mse,
                "coverage": cov,
                "n_rep_used": n_used,
                "mean_Nj": float(np.mean(Nj_mat[ok, j])),
                "min_Nj": float(np.min(Nj_mat[ok, j])),
            })

    return rows


# ============================================================
# 8) Grid runner
# ============================================================

def run_experiment_grid(cfg: ExperimentConfig) -> Any:
    """
    Runs the full scenario grid and returns results.

    Returns:
      - pandas DataFrame if pandas is available
      - otherwise returns a list of dict rows.
    """
    scenarios: List[Scenario] = []
    for B in (5, 30):
        for n in (1000, ): 
            for snr in (0.3, 0.2, 0.1):
                for rho in (0.3, 0.6, 0.8):
                    scenarios.append(Scenario(B=B, n=n, snr=snr, dep=DependenceSpec(icc=0.20, rho=rho)))

    all_rows: List[Dict[str, Any]] = []
    for k, scen in enumerate(scenarios, start=1):
        print(f"\n[Grid] Scenario {k}/{len(scenarios)}: B={scen.B}, n={scen.n}, SNR={scen.snr}, ICC={scen.dep.icc}, rho={scen.dep.rho}")
        rows = evaluate_scenario(scen, cfg)
        all_rows.extend(rows)

    # Optional pandas output
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(all_rows)
        return df
    except Exception:
        return all_rows


def save_results_csv(results: Any, path: str) -> None:
    """
    Save results to CSV (works for pandas DataFrame or list-of-dicts).
    """
    try:
        import pandas as pd  # type: ignore
        if hasattr(results, "to_csv"):
            results.to_csv(path, index=False)
            return
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
        return
    except Exception:
        # fallback: simple csv writer for list-of-dicts
        import csv
        if not isinstance(results, list) or len(results) == 0:
            raise ValueError("results must be a non-empty list of dicts to write without pandas")
        fieldnames = list(results[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in results:
                w.writerow(row)

def summarize_over_j_for_paper(
    df,
    *,
    j_min: int = 2,
) :
    """
    Paper helper #1:
    Aggregate metrics over j >= j_min within each scenario and method.

    Produces scenario-level summaries such as:
      - avg_coverage over j>=2
      - min_coverage over j>=2
      - avg_sd, avg_mse, etc.

    Returns a pandas DataFrame.
    """
    import pandas as pd  # type: ignore

    required = {"B", "n", "snr", "icc", "rho", "method", "j", "sd", "bias", "mse", "coverage", "n_rep_used"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    d = df.copy()
    d = d[d["j"] >= j_min].copy()

    # Make sure metrics are numeric
    for col in ["sd", "bias", "mse", "coverage", "n_rep_used", "mean_Nj", "min_Nj"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Group by scenario and method
    gcols = ["B", "n", "snr", "icc", "rho", "method"]

    # Helper aggregations that ignore NaNs
    def _nanmean(x):
        return float(np.nanmean(x.values)) if np.isfinite(np.nanmean(x.values)) else np.nan

    def _nanmin(x):
        v = x.values
        v = v[np.isfinite(v)]
        return float(np.min(v)) if v.size > 0 else np.nan

    def _nanmax(x):
        v = x.values
        v = v[np.isfinite(v)]
        return float(np.max(v)) if v.size > 0 else np.nan

    summary = (
        d.groupby(gcols, as_index=False)
        .agg(
            j_count=("j", "count"),
            avg_sd=("sd", _nanmean),
            avg_bias=("bias", _nanmean),
            avg_abs_bias=("bias", lambda x: float(np.nanmean(np.abs(x.values)))),
            avg_mse=("mse", _nanmean),
            avg_coverage=("coverage", _nanmean),
            min_coverage=("coverage", _nanmin),
            max_coverage=("coverage", _nanmax),
            avg_mean_Nj=("mean_Nj", _nanmean) if "mean_Nj" in d.columns else ("j", "count"),
            min_min_Nj=("min_Nj", _nanmin) if "min_Nj" in d.columns else ("j", "count"),
            avg_n_rep_used=("n_rep_used", _nanmean),
        )
    )

    # Clean placeholders if columns were missing
    if "mean_Nj" not in d.columns:
        summary = summary.drop(columns=["avg_mean_Nj"], errors="ignore")
    if "min_Nj" not in d.columns:
        summary = summary.drop(columns=["min_min_Nj"], errors="ignore")

    return summary


def _scenario_key_row(row) -> str:
    """Helper to build a stable scenario key string for filenames/titles."""
    return f"B={int(row['B'])}_n={int(row['n'])}_snr={row['snr']}_icc={row['icc']}_rho={row['rho']}"


def plot_metric_vs_j_by_scenario(
    df,
    *,
    metric: str,
    output_dir: str,
    alpha: float = 0.05,
    scenarios_limit: Optional[int] = None,
) -> None:
    """
    Paper helper #2:
    For each scenario (B,n,snr,icc,rho), plot metric vs j with both methods overlaid.
    Saves one PNG per scenario per metric.

    metric in {"sd","mse","coverage","bias"}.
    """
    import os
    import pandas as pd  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    os.makedirs(output_dir, exist_ok=True)

    required = {"B", "n", "snr", "icc", "rho", "method", "j", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    # Sort for nicer plots
    d = df.copy()
    d = d.sort_values(["B", "n", "snr", "icc", "rho", "method", "j"])

    scen_cols = ["B", "n", "snr", "icc", "rho"]
    scenarios = (
        d[scen_cols]
        .drop_duplicates()
        .sort_values(scen_cols)
        .reset_index(drop=True)
    )

    if scenarios_limit is not None:
        scenarios = scenarios.iloc[:scenarios_limit]

    # For each scenario, plot both methods
    for _, scen in scenarios.iterrows():
        mask = (
            (d["B"] == scen["B"])
            & (d["n"] == scen["n"])
            & (d["snr"] == scen["snr"])
            & (d["icc"] == scen["icc"])
            & (d["rho"] == scen["rho"])
        )
        ds = d.loc[mask].copy()
        if ds.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.0, 4.2))

        for method in ["ours", "naive"]:
            dm = ds[ds["method"] == method]
            if dm.empty:
                continue
            ax.plot(dm["j"].values, dm[metric].values, marker="o", label=method)

        # Helpful reference lines
        if metric == "coverage":
            ax.axhline(1.0 - alpha, linewidth=1.0)
            ax.set_ylim(0.0, 1.02)

        ax.set_xlabel("Exposure index j")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs j | { _scenario_key_row(scen) }")
        ax.legend()
        fig.tight_layout()

        fname = f"{metric}__{_scenario_key_row(scen)}.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=160)
        plt.close(fig)


def make_paper_outputs(
    results_df,
    *,
    output_dir: str = "paper_outputs",
    figures_dir: str = "paper_outputs/figures",
    alpha: float = 0.05,
    scenarios_limit_for_plots: Optional[int] = None,
    make_all_plots: bool = True,
) -> None:
    """
    Convenience wrapper:
      - writes scenario summary CSV
      - optionally writes plots (sd/mse/coverage vs j)

    results_df must be the DataFrame returned by run_experiment_grid().
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 1) Scenario-level summary table (aggregated over j>=2)
    summary = summarize_over_j_for_paper(results_df, j_min=2)
    summary_path = os.path.join(output_dir, "scenario_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[Paper] Wrote scenario summary: {summary_path}")

    # 2) Per-scenario plots
    if make_all_plots:
        os.makedirs(figures_dir, exist_ok=True)
        for metric in ["sd", "mse", "coverage"]:
            plot_metric_vs_j_by_scenario(
                results_df,
                metric=metric,
                output_dir=figures_dir,
                alpha=alpha,
                scenarios_limit=scenarios_limit_for_plots,
            )
        print(f"[Paper] Wrote figures to: {figures_dir}")

# ============================================================
# 9) Main
# ============================================================

def main() -> None:
    cfg = ExperimentConfig(
        R_mc=200,
        R_boot=200,
        alpha=0.05,
        seed=20260306,
        min_support=5,
        solver="lstsq",
        ridge=0.0,
    )

    results = run_experiment_grid(cfg)

    # Save raw results
    try:
        import pandas as pd  # type: ignore
        df = results  # type: ignore
        save_results_csv(df, "experiment_results.csv")
        print("\nSaved: experiment_results.csv")

        # --- Paper helpers ---
        # For a quick preview: limit to first 6 scenarios for plots.
        # Set scenarios_limit_for_plots=None to plot all scenarios (can create 108 figures).
        make_paper_outputs(
            df,
            output_dir="paper_outputs",
            figures_dir="paper_outputs/figures",
            alpha=cfg.alpha,
            scenarios_limit_for_plots=6,   # change to None for all scenarios
            make_all_plots=True,
        )

    except Exception:
        # If pandas isn't installed, still save raw CSV, but paper helpers need pandas/matplotlib.
        save_results_csv(results, "experiment_results.csv")
        print("\nSaved: experiment_results.csv")
        print("[Paper] Skipped paper outputs because pandas/matplotlib were unavailable.")


if __name__ == "__main__":
    main()