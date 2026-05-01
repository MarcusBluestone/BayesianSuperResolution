import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIG  — edit results_dir to point at your run
# ============================================================
results_dir = Path("results")
plot_dir = results_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD
# ============================================================
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

true_vals = load_json(results_dir / "data" / "true_values.json")

models = {
    "Bayesian":  load_json(results_dir / "bayes"      / "patch" / "learned_params.json"),
    "MAP (full)": load_json(results_dir / "map_full"  / "full"  / "learned_params.json"),
    "MAP (patch)": load_json(results_dir / "map_patch" / "patch" / "learned_params.json"),
}

colors = {
    "Bayesian":   "steelblue",
    "MAP (full)": "darkorange",
    "MAP (patch)": "seagreen",
}

shifts_true = np.array(true_vals["shifts"])   # (K, 2)
rots_true   = np.array(true_vals["rots"])     # (K,)
K = len(rots_true)
idx = np.arange(K)


# ============================================================
# HELPERS
# ============================================================
def to_relative(shifts: np.ndarray, rots: np.ndarray):
    """Anchor image-0 to zero so absolute offsets don't matter."""
    return shifts - shifts[0:1], rots - rots[0]


shifts_true_rel, rots_true_rel = to_relative(shifts_true, rots_true)

parsed = {}
for name, params in models.items():
    s = np.array(params["shifts"])
    r = np.array(params["rots"])
    s_rel, r_rel = to_relative(s, r)
    parsed[name] = dict(
        shifts_rel=s_rel,
        rots_rel=r_rel,
        gamma=float(params["gamma"]),
        shift_err=np.linalg.norm(s_rel - shifts_true_rel, axis=1),
        rot_err_deg=np.abs(np.degrees(r_rel - rots_true_rel)),
    )


# ============================================================
# PLOT 1: 2-D shift scatter
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    shifts_true_rel[:, 0], shifts_true_rel[:, 1],
    marker="x", s=120, color="black", zorder=5, label="Ground truth",
)

for name, d in parsed.items():
    ax.scatter(
        d["shifts_rel"][:, 0], d["shifts_rel"][:, 1],
        color=colors[name], label=name, zorder=4,
    )
    for i in range(K):
        ax.plot(
            [shifts_true_rel[i, 0], d["shifts_rel"][i, 0]],
            [shifts_true_rel[i, 1], d["shifts_rel"][i, 1]],
            color=colors[name], alpha=0.25, linewidth=0.9,
        )

ax.set_xlabel("Shift x", fontsize=16)
ax.set_ylabel("Shift y", fontsize=16)
ax.set_title("Shift estimation", fontsize=18)
ax.legend(fontsize=12)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(plot_dir / "shift_2d_comparison.png", dpi=150)
plt.close(fig)
print("Saved shift_2d_comparison.png")


# ============================================================
# PLOT 2: rotation absolute error (degrees) — grouped bar
# ============================================================
n_models = len(parsed)
bar_width = 0.8 / n_models
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width

fig, ax = plt.subplots(figsize=(13, 5))

for offset, (name, d) in zip(offsets, parsed.items()):
    ax.bar(idx + offset, d["rot_err_deg"], width=bar_width,
           color=colors[name], label=name)

ax.set_xlabel("Observation index", fontsize=14)
ax.set_ylabel("Absolute rotation error (degrees)", fontsize=14)
ax.set_title("Rotation absolute error by observation", fontsize=16)
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig(plot_dir / "rotation_abs_error_deg.png", dpi=150)
plt.close(fig)
print("Saved rotation_abs_error_deg.png")


# ============================================================
# PLOT 3: per-observation shift error magnitude
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5))

for offset, (name, d) in zip(offsets, parsed.items()):
    ax.bar(idx + offset, d["shift_err"], width=bar_width,
           color=colors[name], label=name)

ax.set_xlabel("Observation index", fontsize=14)
ax.set_ylabel("Shift error (HR pixels)", fontsize=14)
ax.set_title("Shift error magnitude by observation", fontsize=16)
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig(plot_dir / "shift_abs_error.png", dpi=150)
plt.close(fig)
print("Saved shift_abs_error.png")


# ============================================================
# PLOT 4: gamma comparison (horizontal bar)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 3))

names = list(parsed.keys())
gammas = [parsed[n]["gamma"] for n in names]
bar_colors = [colors[n] for n in names]

bars = ax.barh(names, gammas, color=bar_colors, height=0.5)
ax.axvline(x=true_vals["gamma"], color="black", linestyle="--",
           linewidth=1.5, label=f"True γ = {true_vals['gamma']}")
ax.set_xlabel("Learned γ", fontsize=14)
ax.set_xlim(0, 4)
ax.set_title("PSF width (γ) estimation", fontsize=16)
ax.legend(fontsize=11)
for bar, val in zip(bars, gammas):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=11)
fig.tight_layout()
fig.savefig(plot_dir / "gamma_comparison.png", dpi=150)
plt.close(fig)
print("Saved gamma_comparison.png")


# ============================================================
# SUMMARY TABLE
# ============================================================
print()
print(f"{'Model':<16} {'γ':>7}  {'mean shift err':>16}  {'mean rot err (°)':>18}")
print("-" * 62)
for name, d in parsed.items():
    print(
        f"{name:<16} {d['gamma']:>7.3f}  "
        f"{d['shift_err'].mean():>16.4f}  "
        f"{d['rot_err_deg'].mean():>18.4f}"
    )
print(f"{'True':<16} {true_vals['gamma']:>7.3f}")
print()
print(f"Plots saved to: {plot_dir}")