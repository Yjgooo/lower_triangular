"""Make a reproducible, explicitly simulated project illustration.

Install dependencies and run from the repository root::

    python -m pip install numpy scipy matplotlib
    python plot_exposure_project.py

Recreates the presentation figure with n=1,000 users, B=30 exposure levels,
N_j=floor(n/j), independent Gaussian errors (sigma=0.85), and seed 20260919.
Writes outputs/exposure_aware_ab_test.png beside this script and prints JSON
diagnostics. This is one illustrative simulation, not a convergence study.

The fitted model is y_ij = alpha_{T_i} + b_j + epsilon_ij.
Both fits profile out a separate intercept for every observed history length.
Set b_1=0 for identifiability. The constrained fit has b_1 <= ... <= b_B.
The random seed is fixed in advance; no replicate selection is performed.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(20260919)
n = 1000
B = 30
sigma = 0.85
j = np.arange(1, B + 1)
reach = n // j
counts = reach - np.r_[reach[1:], 0]
truth = np.select([j <= 5, j <= 12, j <= 21], [0., .6, 1.1], default=1.55)

H = np.zeros((B, B))
g = np.zeros(B)
histories = []
for t in range(1, B + 1):
    m = int(counts[t - 1])
    if not m:
        continue
    alpha = .5 * np.log1p(t)
    outcomes = alpha + truth[:t] + rng.normal(0, sigma, size=(m, t))
    centered_sum = (outcomes - outcomes.mean(axis=1, keepdims=True)).sum(axis=0)
    H[:t, :t] += m * (np.eye(t) - np.ones((t, t)) / t)
    g[:t] += centered_sum
    histories.extend([t] * m)

# Profiling one length-specific intercept yields this same H and g;
# centering individual rows is an algebraically equivalent sufficient statistic.
Hr, gr = H[1:, 1:], g[1:]
unrestricted = np.r_[0., np.linalg.solve(Hr, gr)]
R = np.linalg.cholesky(Hr).T
pseudo_y = solve_triangular(R.T, gr, lower=True)
L = np.tril(np.ones((B - 1, B - 1)))
A = R @ L
increments, _ = nnls(A, pseudo_y, maxiter=100 * B)
monotone = np.r_[0., L @ increments]

# Meaningful checks: exact triangular design, normal equations, monotonicity,
# and KKT optimality of the constrained quadratic program.
assert len(histories) == n
assert all(sum(t >= q for t in histories) == reach[q - 1] for q in j)
assert np.max(np.abs(Hr @ unrestricted[1:] - gr)) < 1e-8
assert np.min(np.diff(monotone)) >= -1e-10
gradient = A.T @ (A @ increments - pseudo_y)
assert gradient.min() > -1e-7
assert np.max(np.abs(gradient[increments > 1e-9])) < 1e-7

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.edgecolor": "#CDD5DF",
    "axes.labelcolor": "#36465B",
    "xtick.color": "#556477",
    "ytick.color": "#556477",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})
navy = "#172D45"
teal = "#007F83"
orange = "#D58B4B"
gray = "#9BA8B8"
muted = "#53657B"

fig = plt.figure(figsize=(16, 9), facecolor="white")
fig.text(.055, .928, "Exposure-aware A/B testing", fontsize=27,
         fontweight="bold", color=navy)
fig.text(.055, .877,
         "How does a feature's effect change as users encounter it repeatedly?",
         fontsize=16, color=muted)
fig.text(.945, .929, "SIMULATED EXAMPLE", fontsize=10.5, color=teal,
         ha="right", va="center", fontweight="bold",
         bbox=dict(boxstyle="round,pad=.55", fc="#E8F5F4", ec="none"))

gs = fig.add_gridspec(1, 2, left=.065, right=.952, bottom=.248, top=.75,
                      width_ratios=[.90, 1.75], wspace=.26)
axn = fig.add_subplot(gs[0, 0])
ax = fig.add_subplot(gs[0, 1])

axn.set_title("1   Fewer users reach later exposures", loc="left", pad=30,
              color=navy, fontweight="bold", fontsize=14)
axn.step(j, reach, where="post", color=orange, lw=2.6)
axn.fill_between(j, reach[-1], reach, step="post", color=orange, alpha=.12)
axn.set_yscale("log")
axn.set_ylim(25, 1600)
axn.set_xlim(0, 31)
axn.set_xticks([1, 10, 20, 30])
axn.yaxis.set_major_locator(FixedLocator([33, 100, 300, 1000]))
axn.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
axn.minorticks_off()
axn.set_xlabel("Exposure number", labelpad=12)
axn.set_ylabel("Users observed (log scale)", labelpad=10)
axn.grid(axis="y", color="#E9EDF2", lw=.9)
axn.set_axisbelow(True)
axn.scatter([1, B], [n, reach[-1]], s=44, color=orange, zorder=5)
axn.annotate("1,000 users at\nexposure 1", xy=(1, n), xytext=(8, 900),
             fontsize=11, color=navy, va="center",
             arrowprops=dict(arrowstyle="-", color=orange, lw=1.25,
                             connectionstyle="angle,angleA=180,angleB=85,rad=5"))
axn.annotate("33 users at\nexposure 30", xy=(B, reach[-1]), xytext=(17, 105),
             fontsize=11, color=navy, va="center",
             arrowprops=dict(arrowstyle="-", color=orange, lw=1.25))

ax.set_title("2   Monotonicity shares information across exposures", loc="left",
             pad=30, color=navy, fontweight="bold", fontsize=14)
ax.axhline(0, color="#DAE1E8", lw=1, zorder=0)
line_u, = ax.plot(j, unrestricted, color=gray, lw=1.3, alpha=.88,
                 label="Unrestricted fit", zorder=2)
line_m, = ax.step(j, monotone, where="mid", color=teal, lw=3.1,
                 label="Monotone fit", zorder=4)
line_t, = ax.step(j, truth, where="mid", color=navy, linestyle=(0, (4, 3)), lw=2.0,
                 label="True curve", zorder=5)
ax.set_xlim(0, 31)
ymin = min(unrestricted.min(), monotone.min(), truth.min())
ymax = max(unrestricted.max(), monotone.max(), truth.max())
span = ymax - ymin
ax.set_ylim(ymin - .14 * span, ymax + .10 * span)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("Exposure number", labelpad=12)
ax.set_ylabel("Change from first exposure (outcome units)", labelpad=10)
ax.grid(axis="y", color="#E9EDF2", lw=.9)
ax.set_axisbelow(True)
ax.legend(handles=[line_t, line_u, line_m], loc="upper left",
          bbox_to_anchor=(0, 1.025), ncol=3, frameon=False, fontsize=10.5,
          handlelength=2.5, columnspacing=1.8, borderaxespad=0)

fig.text(.055, .138,
         "Pool information across exposure levels to reduce noise in the estimated curve.",
         fontsize=15, color=navy, fontweight="bold")
fig.text(.055, .096,
         "Single simulated dataset; n = 1,000 users, B = 30 exposures, and Nⱼ = ⌊n/j⌋. Gaussian errors; a monotone truth.",
         fontsize=10.5, color=muted)
fig.text(.055, .065,
         "Both fits adjust for baseline differences by observed history length. The shape constraint is used only for the monotone fit.",
         fontsize=10.5, color=muted)

png = OUT / "exposure_aware_ab_test.png"
fig.savefig(png, dpi=220)
plt.close(fig)
print(json.dumps({
    "file": str(png),
    "n": n, "B": B, "total_observations": int(reach.sum()),
    "final_exposure_users": int(reach[-1]),
    "unrestricted_curve_mse": float(np.mean((unrestricted - truth) ** 2)),
    "monotone_curve_mse": float(np.mean((monotone - truth) ** 2)),
    "truth_endpoint": float(truth[-1]),
    "monotone_endpoint": float(monotone[-1]),
    "unrestricted_range": [float(unrestricted.min()), float(unrestricted.max())],
    "checks": "design, normal equations, monotonicity, KKT passed",
}, indent=2))
