#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np


# ============================================================
# 1) Data container
# ============================================================

@dataclass
class TriangularPanel:
    """
    Triangular trajectories:
      - user i has length T[i] and values Y[i, 1:T[i]]

    Storage:
      - Y_list[i] is a 1D numpy array of shape (T[i],)
      - T is shape (n,), integers in [1, B]
      - B is the max possible length/exposure index

    Convention:
      - math uses 1-based j; Python uses 0-based indexing into arrays.
    """
    Y_list: List[np.ndarray]
    T: np.ndarray
    B: int

    @property
    def n(self) -> int:
        return len(self.Y_list)

    def validate(self) -> "TriangularPanel":
        if not isinstance(self.B, int) or self.B < 1:
            raise ValueError(f"B must be a positive int, got {self.B}")

        n = self.n
        if self.T.shape != (n,):
            raise ValueError(f"T must have shape (n,), got {self.T.shape} with n={n}")
        if not np.issubdtype(self.T.dtype, np.integer):
            raise TypeError(f"T must be integer dtype, got {self.T.dtype}")

        if np.any(self.T < 1) or np.any(self.T > self.B):
            bad = self.T[(self.T < 1) | (self.T > self.B)]
            raise ValueError(f"T has values outside [1, B]. Example: {bad[:10]}")

        for i, (Yi, t) in enumerate(zip(self.Y_list, self.T)):
            t = int(t)
            if not isinstance(Yi, np.ndarray):
                raise TypeError(f"Y_list[{i}] must be a numpy array")
            if Yi.ndim != 1 or Yi.shape[0] != t:
                raise ValueError(
                    f"Y_list[{i}] must be 1D with length T[i]={t}, got shape {Yi.shape}"
                )
            if not np.issubdtype(Yi.dtype, np.floating) and not np.issubdtype(Yi.dtype, np.integer):
                raise TypeError(f"Y_list[{i}] dtype must be numeric, got {Yi.dtype}")

        return self


# ============================================================
# 2) Simulation: Option A
# ============================================================

@dataclass(frozen=True)
class OptionAParams:
    """
    Parameters for Option A:

      Y_ij = a_true(T_i) + b_true(j) + u_i + eps_ij

      u_i correlated with latent A_i which influences T_i
      eps_ij AR(1) with heteroskedastic innovations
    """
    B: int
    lam0: float = 2.5
    gamma: float = 0.9
    corr_u_A: float = 0.3
    tau_u: float = 0.5
    rho: float = 0.6
    sigma0: float = 1.0
    sigma_decay: float = 0.04

    def validate(self) -> "OptionAParams":
        if self.B < 1:
            raise ValueError("B must be >= 1")
        if self.lam0 <= 0:
            raise ValueError("lam0 must be > 0")
        if not (-1.0 <= self.corr_u_A <= 1.0):
            raise ValueError("corr_u_A must be in [-1, 1]")
        if self.tau_u < 0:
            raise ValueError("tau_u must be >= 0")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError("rho must be in (-1, 1) for stability")
        if self.sigma0 <= 0:
            raise ValueError("sigma0 must be > 0")
        if self.sigma_decay < 0:
            raise ValueError("sigma_decay must be >= 0")
        return self


def default_truth_curves(B: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (a_true, b_true) as 1-based arrays of shape (B+1,).
      - a_true[t] defined for t=1..B
      - b_true[1] = 0
    """
    tgrid = np.arange(1, B + 1)
    a = 0.25 * np.log1p(tgrid)  # length B, corresponds to t=1..B

    jgrid = np.arange(1, B + 1)
    b = -0.6 * (1.0 - np.exp(-(jgrid - 1) / 6.0))  # length B, corresponds to j=1..B
    b[0] = 0.0  # ensure b(1)=0

    a1 = np.full(B + 1, np.nan, dtype=float)
    b1 = np.full(B + 1, np.nan, dtype=float)
    a1[1:] = a
    b1[1:] = b
    return a1, b1


def simulate_option_A(
    n: int,
    params: OptionAParams,
    seed: Optional[Union[int, np.random.SeedSequence]] = None,
    return_latents: bool = False,
) -> Tuple[TriangularPanel, np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Simulate triangular panel under Option A.

    Returns:
      panel, a_true(1-based), b_true(1-based), latents(optional)
    """
    params.validate()
    rng = np.random.default_rng(seed)
    B = params.B

    a_true, b_true = default_truth_curves(B)

    # Latent A and random effect u with desired correlation corr_u_A
    # A has variance 1, u has variance tau_u^2, cov = corr * 1 * tau_u
    cov = params.corr_u_A * params.tau_u
    Sigma = np.array([[1.0, cov], [cov, params.tau_u ** 2]], dtype=float)

    # Numerical check: Sigma PSD (should be if |corr|<=1)
    # If tau_u=0, Sigma is singular but still fine.
    try:
        L = np.linalg.cholesky(Sigma + 1e-15 * np.eye(2))
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Covariance not PSD. Sigma=\n{Sigma}") from e

    z = rng.standard_normal(size=(n, 2))
    Au = z @ L.T
    A = Au[:, 0]
    u = Au[:, 1]

    # Generate lengths T
    lam = params.lam0 * np.exp(params.gamma * A)
    T = 1 + rng.poisson(lam=lam)
    T = np.clip(T, 1, B).astype(int)

    # Generate outcomes with AR(1) within user
    Y_list: List[np.ndarray] = []
    for i in range(n):
        t = int(T[i])
        Yi = np.empty(t, dtype=float)
        eps_prev = 0.0  # non-stationary start (fine; mean is still 0)

        for jj in range(1, t + 1):
            sigma_eta = params.sigma0 * math.exp(-params.sigma_decay * (jj - 1))
            eta = rng.normal(0.0, sigma_eta)
            eps = params.rho * eps_prev + eta
            eps_prev = eps

            mean = a_true[t] + b_true[jj] + u[i]
            Yi[jj - 1] = mean + eps

        Y_list.append(Yi)

    panel = TriangularPanel(Y_list=Y_list, T=T, B=B).validate()

    latents = None
    if return_latents:
        latents = {"A": A, "u": u, "T": T}

    return panel, a_true, b_true, latents


# ============================================================
# 3) Estimation: exact WLS with b(1)=0 via normal equations
# ============================================================

@dataclass(frozen=True)
class AdditiveTriangularFit:
    """
    Fit result for:
      E[Y_ij | T=t] = a(t) + b(j), j<=t, with b(1)=0

    a and b are 1-based arrays of shape (B+1,):
      - a[0] = NaN, b[0] = NaN
      - b[1] = 0 by construction
      - entries can be NaN if not identifiable (no support)
    """
    a: np.ndarray
    b: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class _SuffStats:
    B: int
    n_users: int
    n_obs: int
    S_t: np.ndarray     # sum of weights by length t
    Ysum_t: np.ndarray  # sum of weights * sum Y by length t
    W_j: np.ndarray     # sum of weights among users with T>=j
    Ysum_j: np.ndarray  # sum of weights * Y at exposure j among users with T>=j


def _compute_suffstats(panel: TriangularPanel, weights: Optional[np.ndarray]) -> _SuffStats:
    panel.validate()
    n, B = panel.n, panel.B

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError(f"weights must have shape (n,), got {w.shape}")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")
    n_obs = int(np.sum(panel.T))

    # S_t[t] = sum_{i: T_i=t} w_i
    S_t = np.bincount(panel.T, weights=w, minlength=B + 1).astype(float)

    Ysum_t = np.zeros(B + 1, dtype=float)
    Ysum_j = np.zeros(B + 1, dtype=float)

    for i in range(n):
        t = int(panel.T[i])
        wi = float(w[i])
        Yi = panel.Y_list[i]

        Ysum_t[t] += wi * float(np.sum(Yi))
        Ysum_j[1 : t + 1] += wi * Yi

    # W_j[j] = sum_{t>=j} S_t[t]
    W_j = np.cumsum(S_t[::-1])[::-1]

    return _SuffStats(B=B, n_users=n, n_obs=n_obs, S_t=S_t, Ysum_t=Ysum_t, W_j=W_j, Ysum_j=Ysum_j)


def fit_additive_triangular_wls(
    panel: TriangularPanel,
    weights: Optional[np.ndarray] = None,
    *,
    ridge: float = 0.0,
    solver: str = "lstsq",
    compute_objective: bool = False,
) -> AdditiveTriangularFit:
    """
    Exact weighted least squares fit with constraint b(1)=0:

      minimize  sum_i sum_{j=1..T_i} w_i (Y_ij - a(T_i) - b(j))^2
      subject to b(1)=0

    Approach:
      - Build normal equations using sufficient statistics (no iteration)
      - Solve small system of size <= (2B-1)

    solver:
      - "lstsq": robust (default)
      - "solve": strict solve (fails if singular)
    """
    if ridge < 0:
        raise ValueError("ridge must be >= 0")

    stats = _compute_suffstats(panel, weights)
    B = stats.B

    # Identify supported parameters
    active_t = np.array([t for t in range(1, B + 1) if stats.S_t[t] > 0], dtype=int)
    active_j = np.array([j for j in range(2, B + 1) if stats.W_j[j] > 0], dtype=int)

    p = active_t.size + active_j.size
    if p == 0:
        raise ValueError("No identifiable parameters under given weights/data.")

    idx_a = {t: k for k, t in enumerate(active_t)}
    idx_b = {j: active_t.size + k for k, j in enumerate(active_j)}

    M = np.zeros((p, p), dtype=float)
    r = np.zeros(p, dtype=float)

    # a-block and RHS
    for t in active_t:
        ia = idx_a[t]
        M[ia, ia] = (t * stats.S_t[t]) + ridge
        r[ia] = stats.Ysum_t[t]

    # b-block and RHS
    for j in active_j:
        ib = idx_b[j]
        M[ib, ib] = stats.W_j[j] + ridge
        r[ib] = stats.Ysum_j[j]

    # cross terms: for each t, all j<=t couple with mass S_t[t]
    for t in active_t:
        ia = idx_a[t]
        mass = stats.S_t[t]
        for j in active_j:
            if j <= t:
                ib = idx_b[j]
                M[ia, ib] = mass
                M[ib, ia] = mass

    # Solve
    diag: Dict[str, Any] = {"solver": solver, "ridge": ridge, "active_t": active_t, "active_j": active_j}
    if solver == "solve":
        theta = np.linalg.solve(M, r)
        diag["rank"] = int(np.linalg.matrix_rank(M))
        diag["residual_norm"] = float(np.linalg.norm(M @ theta - r))
    elif solver == "lstsq":
        theta, residuals, rank, svals = np.linalg.lstsq(M, r, rcond=None)
        diag["rank"] = int(rank)
        diag["singular_values"] = svals
        diag["residual_norm"] = float(np.linalg.norm(M @ theta - r))
        if svals.size >= 2 and svals[-1] > 0:
            diag["cond"] = float(svals[0] / svals[-1])
        else:
            diag["cond"] = float("inf")
        if residuals is not None and np.size(residuals) > 0:
            diag["lstsq_residual_sum_squares"] = float(residuals[0])
    else:
        raise ValueError("solver must be 'solve' or 'lstsq'")

    # Unpack into 1-based arrays
    a = np.full(B + 1, np.nan, dtype=float)
    b = np.full(B + 1, np.nan, dtype=float)
    b[1] = 0.0  # constraint

    for k, t in enumerate(active_t):
        a[t] = theta[k]
    offset = active_t.size
    for k, j in enumerate(active_j):
        b[j] = theta[offset + k]

    if compute_objective:
        if weights is None:
            w = np.ones(panel.n, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
        obj = 0.0
        for i in range(panel.n):
            t = int(panel.T[i])
            wi = float(w[i])
            Yi = panel.Y_list[i]
            pred = a[t] + b[1 : t + 1]
            resid = Yi - pred
            obj += wi * float(np.sum(resid * resid))
        diag["objective"] = obj

    return AdditiveTriangularFit(a=a, b=b, diagnostics=diag)


# ============================================================
# 4) Naive baseline estimator (fixed)
# ============================================================

def naive_b_curve(panel: TriangularPanel, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Naive estimator of b(j): mean of Y_ij at each exposure j among those observed at j,
    then shifts so b(1)=0.

    Returns b as 1-based array of shape (B+1,) with b[1]=0.
    """
    panel.validate()
    n, B = panel.n, panel.B

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must have shape (n,)")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")

    num = np.zeros(B + 1, dtype=float)
    den = np.zeros(B + 1, dtype=float)

    for i in range(n):
        t = int(panel.T[i])
        wi = float(w[i])
        Yi = panel.Y_list[i]
        num[1 : t + 1] += wi * Yi
        den[1 : t + 1] += wi

    b = np.full(B + 1, np.nan, dtype=float)
    # safe division
    np.divide(num[1:], den[1:], out=b[1:], where=(den[1:] > 0))

    if not np.isfinite(b[1]):
        raise ValueError("b(1) is not identifiable (no weight mass at j=1). Check weights/data.")

    # enforce b(1)=0
    b[1:] -= b[1]
    b[1] = 0.0
    return b


# ============================================================
# 5) Cluster bootstrap (Poisson(1) user weights)
# ============================================================

def bootstrap_b_curve(
    panel: TriangularPanel,
    *,
    R: int = 300,
    seed: Optional[Union[int, np.random.SeedSequence]] = None,
    base_weights: Optional[np.ndarray] = None,
    solver: str = "lstsq",
    ridge: float = 0.0,
) -> np.ndarray:
    """
    User-cluster bootstrap for b(j) using Poisson(1) weights:
      m_i ~ Poisson(1), independent across users
      effective weight = base_weights[i] * m_i   (or just m_i if base_weights None)

    Returns:
      b_boot: (R, B+1) array; row r is bootstrap b_hat^{*(r)}.
    """
    panel.validate()
    rng = np.random.default_rng(seed)

    n, B = panel.n, panel.B
    if R < 1:
        raise ValueError("R must be >= 1")

    if base_weights is None:
        base = np.ones(n, dtype=float)
    else:
        base = np.asarray(base_weights, dtype=float)
        if base.shape != (n,):
            raise ValueError("base_weights must have shape (n,)")
        if np.any(base < 0):
            raise ValueError("base_weights must be nonnegative")

    out = np.full((R, B + 1), np.nan, dtype=float)

    for r in range(R):
        m = rng.poisson(1.0, size=n).astype(float)
        w = base * m
        if w.sum() <= 0:
            # extremely rare, but handle deterministically
            w[rng.integers(0, n)] = 1.0

        fit = fit_additive_triangular_wls(panel, weights=w, solver=solver, ridge=ridge, compute_objective=False)
        out[r, :] = fit.b

    return out


# ============================================================
# 6) Experiment runner
# ============================================================

@dataclass(frozen=True)
class ExperimentSummary:
    b_true: np.ndarray
    b_hat: np.ndarray
    b_naive: np.ndarray
    b_ci_lo: np.ndarray
    b_ci_hi: np.ndarray
    diagnostics: Dict[str, Any]


def run_single_experiment(
    *,
    n: int,
    params: OptionAParams,
    seed_sim: Optional[Union[int, np.random.SeedSequence]] = None,
    seed_boot: Optional[Union[int, np.random.SeedSequence]] = None,
    R_boot: int = 300,
) -> ExperimentSummary:
    """
    Simulate once, fit b via exact WLS, compute naive baseline, bootstrap CI.
    """
    panel, _a_true, b_true, _ = simulate_option_A(n=n, params=params, seed=seed_sim, return_latents=False)

    fit = fit_additive_triangular_wls(panel, solver="lstsq", ridge=0.0, compute_objective=False)
    b_hat = fit.b

    b_naive = naive_b_curve(panel)

    b_boot = bootstrap_b_curve(panel, R=R_boot, seed=seed_boot)
    b_ci_lo = np.nanquantile(b_boot, 0.025, axis=0)
    b_ci_hi = np.nanquantile(b_boot, 0.975, axis=0)

    diag = dict(fit.diagnostics)
    diag["n"] = n
    diag["R_boot"] = R_boot

    return ExperimentSummary(
        b_true=b_true,
        b_hat=b_hat,
        b_naive=b_naive,
        b_ci_lo=b_ci_lo,
        b_ci_hi=b_ci_hi,
        diagnostics=diag,
    )


def print_experiment_table(summary: ExperimentSummary, max_j: int = 15) -> None:
    B = len(summary.b_true) - 1
    J = min(B, max_j)
    print("j  b_true    b_hat   b_naive   CI_low   CI_high   bias_hat  bias_naive")
    for j in range(1, J + 1):
        bt = summary.b_true[j]
        bh = summary.b_hat[j]
        bn = summary.b_naive[j]
        lo = summary.b_ci_lo[j]
        hi = summary.b_ci_hi[j]
        bias_h = bh - bt
        bias_n = bn - bt
        print(
            f"{j:2d} {bt:8.4f} {bh:8.4f} {bn:8.4f} {lo:8.4f} {hi:8.4f} {bias_h:9.5f} {bias_n:10.5f}"
        )


# ============================================================
# 7) Monte Carlo unbiasedness harness (summary of bias)
# ============================================================

def mc_unbiasedness_option_A(
    *,
    R: int,
    n: int,
    params: OptionAParams,
    seed: int = 12345,
    report_every: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo check: run R replications and compute
      bias_j = mean(b_hat(j)) - b_true(j)
    along with Monte Carlo SE and z-scores.

    Returns a dict of arrays (1-based indexing for b arrays).
    """
    params.validate()
    B = params.B

    ss = np.random.SeedSequence(seed)
    seeds = ss.spawn(R)

    b_hat_mat = np.full((R, B + 1), np.nan, dtype=float)
    b_true_ref: Optional[np.ndarray] = None

    for r in range(R):
        if report_every and ((r + 1) % report_every == 0):
            print(f"[MC] {r+1}/{R}")

        panel, _a_true, b_true, _ = simulate_option_A(n=n, params=params, seed=seeds[r])
        if b_true_ref is None:
            b_true_ref = b_true.copy()

        fit = fit_additive_triangular_wls(panel, solver="lstsq", ridge=0.0, compute_objective=False)
        b_hat_mat[r, :] = fit.b

    assert b_true_ref is not None

    mean_hat = np.nanmean(b_hat_mat, axis=0)
    sd_hat = np.nanstd(b_hat_mat, axis=0, ddof=1)

    mc_se = sd_hat / math.sqrt(R)
    bias = mean_hat - b_true_ref

    z = np.full(B + 1, np.nan, dtype=float)
    ok = np.isfinite(mc_se) & (mc_se > 0)
    z[ok] = bias[ok] / mc_se[ok]

    return {
        "b_true": b_true_ref,
        "mean_b_hat": mean_hat,
        "bias": bias,
        "mc_se": mc_se,
        "z": z,
    }


# ============================================================
# 8) Optional plotting helpers (lazy import)
# ============================================================

def _get_plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:
        raise ImportError("matplotlib required for plotting: pip install matplotlib") from e
    return plt


def plot_b_curves(
    b_true: np.ndarray,
    b_hat: np.ndarray,
    b_ci_lo: Optional[np.ndarray] = None,
    b_ci_hi: Optional[np.ndarray] = None,
    title: str = "b(j) novelty curve",
) -> None:
    plt = _get_plt()
    B = len(b_true) - 1
    j = np.arange(1, B + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(j, b_true[1:], label="b_true(j)")
    ax.plot(j, b_hat[1:], label="b_hat(j)")

    if b_ci_lo is not None and b_ci_hi is not None:
        ax.fill_between(j, b_ci_lo[1:], b_ci_hi[1:], alpha=0.2, label="95% bootstrap CI")

    ax.axhline(0.0, lw=0.8)
    ax.set_xlabel("Exposure index j")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()


# ============================================================
# 9) Demo main (optional)
# ============================================================

def main_demo() -> None:
    params = OptionAParams(B=5, sigma_decay=0.05, corr_u_A=0.0, rho=0.7, sigma0=1.2, lam0=2.0, gamma=0.9, tau_u=0.6)
    summary = run_single_experiment(n=200_000, params=params, seed_sim=1, seed_boot=2, R_boot=300)
    print_experiment_table(summary, max_j=15)
    plot_b_curves(summary.b_true, summary.b_hat, summary.b_ci_lo, summary.b_ci_hi)

    # Monte Carlo bias check (smaller n for speed)
    mc = mc_unbiasedness_option_A(R=500, n=20_000, params=params, seed=123)
    print("\n[MC] bias:", mc["bias"][1:])
    print("[MC] z:", mc["z"][1:])


if __name__ == "__main__":
    main_demo()