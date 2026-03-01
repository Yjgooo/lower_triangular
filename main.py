#!/usr/bin/env python3
"""
Triangular panel additive model:
  E[Y_{ij} | T_i=t] = a(t) + b(j),   1<=j<=t<=B

Estimator:
  (a_hat, b_hat) = argmin_{a,b} sum_i sum_{j<=T_i} (Y_ij - a(T_i) - b(j))^2

We enforce identifiability with: b(1)=0.
Then a(t) absorbs the overall level.

Simulation Option A:
  Y_ij = a_true(T_i) + b_true(j) + u_i + eps_ij
  u_i ~ N(0, tau^2)
  eps_ij AR(1): eps_ij = rho eps_i,j-1 + eta_ij
  eta_ij ~ N(0, sigma_eta(j)^2) with sigma_eta(j) decreasing in j

Inference:
  user-cluster bootstrap: resample users with replacement, refit, collect b_hat.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# -----------------------------
# Data container
# -----------------------------

@dataclass
class TriangularPanel:
    """Stores triangular trajectories: user i has length T[i] and values Y[i,1:T[i]].
    We store as Python lists of 1D numpy arrays.
    """
    Y_list: List[np.ndarray]  # each shape (T_i,)
    T: np.ndarray             # shape (n,), ints in [1,B]
    B: int

    @property
    def n(self) -> int:
        return len(self.Y_list)


# -----------------------------
# Estimator
# -----------------------------

def fit_additive_triangular(
    panel: TriangularPanel,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-10,
    ridge: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a(t), b(j) on triangular grid by weighted least squares with constraint b(1)=0.

    We use alternating conditional least squares (Gauss-Seidel style):
      a(t) = weighted mean over users with T_i=t of (Y_ij - b(j))
      b(j) = weighted mean over users with T_i>=j of (Y_ij - a(T_i))
    and then set b(1)=0 by shifting:
      delta = b(1);  b <- b - delta;  a <- a + delta

    Args:
      weights: optional per-user weights w_i >=0 (e.g. bootstrap counts). Default all 1.
      ridge: small ridge penalty on a and b (optional). If >0, stabilizes tiny counts.

    Returns:
      a_hat: shape (B+1,) with a_hat[t] defined for t=1..B (index 0 unused)
      b_hat: shape (B+1,) with b_hat[j] defined for j=1..B (index 0 unused), and b_hat[1]=0.
    """
    n, B = panel.n, panel.B
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must have shape (n,)")

    # initialize
    a = np.zeros(B + 1, dtype=float)
    b = np.zeros(B + 1, dtype=float)

    # precompute index sets to avoid Python conditionals in the loop
    # users_by_t[t] = list of user indices with T_i=t
    users_by_t: List[List[int]] = [[] for _ in range(B + 1)]
    for i, t in enumerate(panel.T):
        users_by_t[int(t)].append(i)

    # users_ge_j[j] = list of user indices with T_i>=j
    users_ge_j: List[List[int]] = [[] for _ in range(B + 1)]
    # build via cumulative fill
    # O(nB) worst-case; for typical B<=200 it's fine.
    for j in range(1, B + 1):
        users_ge_j[j] = [i for i, t in enumerate(panel.T) if t >= j]

    def update_a(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_new = a.copy()
        for t in range(1, B + 1):
            idx = users_by_t[t]
            if not idx:
                continue
            num = 0.0
            den = 0.0
            # sum over users with T_i=t and their j=1..t observations
            for i in idx:
                wi = w[i]
                Yi = panel.Y_list[i]  # length t
                # residual after subtracting b(j)
                # b indices 1..t correspond to Yi[0..t-1]
                num += wi * float(np.sum(Yi - b[1 : t + 1]))
                den += wi * t
            # ridge: add penalty ridge * a(t)^2 -> adds ridge to denominator, 0 to numerator
            if den <= 0:
    # No effective weight on this row t in this bootstrap replicate
    # Keep previous estimate (or set to 0.0; keeping is usually more stable)
                continue
            a_new[t] = num / (den + ridge)
        return a_new

    def update_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        b_new = b.copy()
        for j in range(1, B + 1):
            idx = users_ge_j[j]
            if not idx:
                continue
            num = 0.0
            den = 0.0
            for i in idx:
                t = int(panel.T[i])
                wi = w[i]
                # Y_{ij} is Yi[j-1]
                yij = float(panel.Y_list[i][j - 1])
                num += wi * (yij - a[t])
                den += wi
            if den <= 0:
                continue
            b_new[j] = num / (den + ridge)
        return b_new

    prev_obj = np.inf
    for it in range(max_iter):
        a = update_a(a, b)
        b = update_b(a, b)

        # enforce b(1)=0 identifiability
        delta = b[1]
        b[1:] -= delta
        a[1:] += delta

        # compute objective occasionally for convergence check
        if it % 5 == 0 or it == max_iter - 1:
            obj = 0.0
            for i in range(n):
                t = int(panel.T[i])
                wi = w[i]
                Yi = panel.Y_list[i]
                pred = a[t] + b[1 : t + 1]
                resid = Yi - pred
                obj += wi * float(np.sum(resid * resid))
            if abs(prev_obj - obj) <= tol * (1.0 + prev_obj):
                break
            prev_obj = obj

    return a, b


# -----------------------------
# Naive mean-by-exposure estimator
# -----------------------------

def naive_b_curve(
    panel: TriangularPanel,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Naive estimator of b(j): average outcomes at each exposure index j, ignoring a(t).

    Uses optional per-user weights; aligns identifiability by shifting so b(1)=0.
    """
    n, B = panel.n, panel.B
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError("weights must have shape (n,)")

    num = np.zeros(B + 1, dtype=float)
    den = np.zeros(B + 1, dtype=float)

    # accumulate weighted sums for each exposure position j
    for i in range(n):
        wi = w[i]
        Yi = panel.Y_list[i]
        t = int(panel.T[i])
        num[1 : t + 1] += wi * Yi
        den[1 : t + 1] += wi

    b = np.zeros(B + 1, dtype=float)
    positive = den[1:] > 0
    if np.any(positive):
        b[1:][positive] = num[1:][positive] / den[1:][positive]

    # enforce b(1)=0 by shifting
    delta = b[1]
    b[1:] -= delta
    return b


# -----------------------------
# Simulation: Option A
# -----------------------------

def simulate_option_A(
    n: int,
    B: int,
    seed: int = 0,
    # exposure count model
    lam0: float = 2.5,
    gamma: float = 0.9,
    # correlation between baseline and activity
    corr_u_A: float = 0.3,
    tau_u: float = 0.5,
    # within-user AR(1)
    rho: float = 0.6,
    sigma0: float = 1.0,
    sigma_decay: float = 0.04,
) -> Tuple[TriangularPanel, np.ndarray, np.ndarray]:
    """
    Simulate triangular panel under Option A.

    True mean:
      a_true(t): user-type-by-exposure-count effect (captures heavy/light baseline)
      b_true(j): learning curve (parallel across user types)

    T_i model:
      A_i ~ N(0,1), T_i = min(B, 1 + Poisson(lam0 * exp(gamma A_i)))

    Correlated (u_i, A_i) to make T informative:
      Corr(u_i, A_i) = corr_u_A, Var(u_i)=tau_u^2.

    Noise:
      eps AR(1) with innovation sd sigma_eta(j) = sigma0 * exp(-sigma_decay*(j-1))
    """
    rng = np.random.default_rng(seed)

    # true curves (you can change these)
    # b_true decreasing (faster over time): e.g. exponential learning benefit
    j = np.arange(1, B + 1)
    b_true = -0.6 * (1.0 - np.exp(-(j - 1) / 6.0))  # starts 0, decreases to about -0.6
    b_true[0] = 0.0  # b(1)=0 for identifiability

    # a_true increasing with t (heavy users slower/faster baseline depending on sign)
    tgrid = np.arange(1, B + 1)
    a_true = 0.25 * np.log1p(tgrid)  # mild trend with t

    # simulate correlated (A_i, u_i)
    # Build covariance matrix for [A, u]
    # Var(A)=1, Var(u)=tau_u^2, Cov = corr * tau_u
    cov = corr_u_A * tau_u
    Sigma = np.array([[1.0, cov], [cov, tau_u**2]])
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal(size=(n, 2))
    Au = z @ L.T
    A = Au[:, 0]
    u = Au[:, 1]

    # simulate T_i
    lam = lam0 * np.exp(gamma * A)
    T = 1 + rng.poisson(lam=lam)
    T = np.clip(T, 1, B).astype(int)

    # simulate trajectories
    Y_list: List[np.ndarray] = []
    for i in range(n):
        t = int(T[i])
        Yi = np.empty(t, dtype=float)
        eps_prev = 0.0
        for jj in range(1, t + 1):
            sigma_eta = sigma0 * math.exp(-sigma_decay * (jj - 1))
            eta = rng.normal(0.0, sigma_eta)
            eps = rho * eps_prev + eta
            eps_prev = eps
            mean = a_true[t - 1] + b_true[jj - 1] + u[i]
            Yi[jj - 1] = mean + eps
        Y_list.append(Yi)

    panel = TriangularPanel(Y_list=Y_list, T=T, B=B)

    # return with 1-indexed arrays for a,b to match fitter output indexing
    a_true_1 = np.zeros(B + 1); a_true_1[1:] = a_true
    b_true_1 = np.zeros(B + 1); b_true_1[1:] = b_true
    return panel, a_true_1, b_true_1


# -----------------------------
# Cluster bootstrap by user
# -----------------------------

def cluster_bootstrap_b(
    panel: TriangularPanel,
    R: int = 300,
    seed: int = 123,
) -> np.ndarray:
    """
    User-cluster bootstrap for b(j):
      resample users with replacement, refit, store b_hat.

    Returns:
      b_boot: shape (R, B+1), b_boot[r, j] = b_hat^{*(r)}(j)
    """
    rng = np.random.default_rng(seed)
    n, B = panel.n, panel.B
    b_boot = np.zeros((R, B + 1), dtype=float)

    # Poisson(1) weighted bootstrap is faster & vectorizes well:
    # weights w_i ~ Poisson(1) iid, equivalent asymptotically to resampling users.
    for r in range(R):
        w = rng.poisson(1.0, size=n).astype(float)
        # if all zero (rare), resample
        if w.sum() == 0:
            w[rng.integers(0, n)] = 1.0
        _, b_hat = fit_additive_triangular(panel, weights=w)
        b_boot[r, :] = b_hat
    return b_boot


# -----------------------------
# Visualization (optional)
# -----------------------------

def _get_plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting; pip install matplotlib") from e
    return plt


def plot_true_curves(
    a_true: np.ndarray,
    b_true: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    tgrid = np.arange(1, len(a_true))
    jgrid = np.arange(1, len(b_true))
    ax.plot(tgrid, a_true[1:], label="a_true(t)", color="C0")
    ax.plot(jgrid, b_true[1:], label="b_true(j)", color="C1")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.set_title("True a(t) and b(j)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_true_mean_surface(
    a_true: np.ndarray,
    b_true: np.ndarray,
    B: int,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    plt = _get_plt()
    grid = np.full((B, B), np.nan, dtype=float)
    for t in range(1, B + 1):
        for j in range(1, t + 1):
            grid[t - 1, j - 1] = a_true[t] + b_true[j]
    masked = np.ma.masked_invalid(grid)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Exposure j")
    ax.set_ylabel("User length t")
    ax.set_title("True mean surface a(t)+b(j)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_true_mean_surface_3d(
    a_true: np.ndarray,
    b_true: np.ndarray,
    B: int,
    save_path: Optional[str] = None,
    show: bool = True,
    elev: float = 25.0,
    azim: float = -135.0,
) -> None:
    """3D surface plot of the true mean a(t)+b(j) over the triangular domain."""
    plt = _get_plt()
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as e:
        raise ImportError("mpl_toolkits.mplot3d is required for 3D plotting") from e

    t_axis = np.arange(1, B + 1)
    j_axis = np.arange(1, B + 1)
    J, T = np.meshgrid(j_axis, t_axis)

    Z = np.full((B, B), np.nan, dtype=float)
    for t in range(1, B + 1):
        for j in range(1, t + 1):
            Z[t - 1, j - 1] = a_true[t] + b_true[j]
    Z_masked = np.ma.masked_invalid(Z)

    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(J, T, Z_masked, cmap="viridis", edgecolor="none", antialiased=True)
    ax.set_xlabel("Exposure j")
    ax.set_ylabel("User length t")
    ax.set_zlabel("Mean a(t)+b(j)")
    ax.set_title("True mean surface (3D)")
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, fraction=0.046, pad=0.08, shrink=0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Demo / sanity check
# -----------------------------

def main():
    n = 20000
    B = 40

    panel, a_true, b_true = simulate_option_A(
        n=n, B=B, seed=0,
        lam0=2.0, gamma=0.9,
        corr_u_A=0.35, tau_u=0.6,
        rho=0.7, sigma0=1.2, sigma_decay=0.05
    )

    a_hat, b_hat = fit_additive_triangular(panel)

    # naive benchmark
    b_naive = naive_b_curve(panel)

    # bootstrap CI for b(j)
    b_boot = cluster_bootstrap_b(panel, R=300, seed=1)
    lo = np.quantile(b_boot, 0.025, axis=0)
    hi = np.quantile(b_boot, 0.975, axis=0)

    # print a small table for b and pointwise errors
    mse_by_j = (b_hat[1:] - b_true[1:]) ** 2
    mse_by_j_naive = (b_naive[1:] - b_true[1:]) ** 2
    print("j  b_true  b_hat  b_naive  CI_low  CI_high  mse_hat  mse_naive")
    for j in range(1, min(B, 15) + 1):
        print(
            f"{j:2d} {b_true[j]:7.3f} {b_hat[j]:7.3f} {b_naive[j]:7.3f} {lo[j]:7.3f} {hi[j]:7.3f} {mse_by_j[j-1]:7.4f} {mse_by_j_naive[j-1]:7.4f}"
        )

    # quick scalar error summaries (ignore identifiability shift handled by b(1)=0)
    mse_b = float(np.mean((b_hat[1:] - b_true[1:])**2))
    mse_b_naive = float(np.mean((b_naive[1:] - b_true[1:])**2))
    print(f"\nMSE(b): {mse_b:.6f}")
    print(f"MSE(b_naive): {mse_b_naive:.6f}")

    # toggle to visualize true curves/surface; kept separate from core logic
    if True:
        plot_true_curves(a_true, b_true)
        plot_true_mean_surface(a_true, b_true, B)
        # 3D view (set to True to enable)
        if True:
            plot_true_mean_surface_3d(a_true, b_true, B)

if __name__ == "__main__":
    main()