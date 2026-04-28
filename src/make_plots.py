import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


results_dir = Path("imgs/test")
plot_dir = results_dir / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Load parameters
# ---------------------------------
params = {}

with open(results_dir / "bayes" / "patch" / "learned_params.json", "r") as f:
    params["bayes"] = json.load(f)

with open(results_dir / "map" / "full" / "learned_params.json", "r") as f:
    params["map"] = json.load(f)

with open(results_dir / "data" / "true_values.json", "r") as f:
    params["true"] = json.load(f)

# Convert to numpy
shifts_true = np.array(params["true"]["shifts"])      # (K, 2)
rots_true = np.array(params["true"]["rots"])          # (K,)

shifts_bayes = np.array(params["bayes"]["shifts"])    # (K, 2)
rots_bayes = np.array(params["bayes"]["rots"])        # (K,)

shifts_map = np.array(params["map"]["shifts"])        # (K, 2)
rots_map = np.array(params["map"]["rots"])            # (K,)

K = len(rots_true)
idx = np.arange(K)

# Colors
bayes_color = "blue"
map_color = "orange"
true_color = "black"

def align_relative(shifts_est, rots_est, shifts_true, rots_true):
    # Anchor both to image 0
    shifts_est_rel = shifts_est - shifts_est[0:1]
    shifts_true_rel = shifts_true - shifts_true[0:1]

    rots_est_rel = rots_est - rots_est[0]
    rots_true_rel = rots_true - rots_true[0]

    return shifts_est_rel, rots_est_rel, shifts_true_rel, rots_true_rel

shifts_bayes_rel, rots_bayes_rel, shifts_true_rel, rots_true_rel = align_relative(
    shifts_bayes, rots_bayes, shifts_true, rots_true
)

shifts_map_rel, rots_map_rel, _, _ = align_relative(
    shifts_map, rots_map, shifts_true, rots_true
)

# ---------------------------------
# SHIFT 2D PLOT WITH ERROR LINES
# ---------------------------------
plt.figure(figsize=(6, 6))

plt.scatter(
    shifts_true_rel[:, 0],
    shifts_true_rel[:, 1],
    marker="x",
    s=100,
    color="black",
    label="Ground truth",
)

plt.scatter(
    shifts_bayes_rel[:, 0],
    shifts_bayes_rel[:, 1],
    color="blue",
    label="Bayesian",
)

plt.scatter(
    shifts_map_rel[:, 0],
    shifts_map_rel[:, 1],
    color="orange",
    label="MAP",
)

for i in range(K):
    plt.plot(
        [shifts_true_rel[i, 0], shifts_bayes_rel[i, 0]],
        [shifts_true_rel[i, 1], shifts_bayes_rel[i, 1]],
        color="gray",
        alpha=0.3,
        linewidth=1,
    )
    plt.plot(
        [shifts_true_rel[i, 0], shifts_map_rel[i, 0]],
        [shifts_true_rel[i, 1], shifts_map_rel[i, 1]],
        color="gray",
        alpha=0.3,
        linewidth=1,
    )

plt.xlabel("Shift x")
plt.ylabel("Shift y")
plt.title("Shift comparison with error vectors")
plt.legend()
plt.axis("equal")  # preserve geometry
plt.tight_layout()
plt.savefig(plot_dir / "shift_2d_comparison.png")
plt.close()

# ---------------------------------
# ROTATION ABSOLUTE ERROR BAR PLOT
# ---------------------------------
rot_err_bayes = np.abs(rots_bayes_rel - rots_true_rel)
rot_err_map = np.abs(rots_map_rel - rots_true_rel)

bar_width = 0.38

plt.figure(figsize=(12, 5))
plt.bar(idx - bar_width / 2, rot_err_bayes, width=bar_width, color=bayes_color, label="Bayesian")
plt.bar(idx + bar_width / 2, rot_err_map, width=bar_width, color=map_color, label="MAP")
plt.xlabel("Observation index")
plt.ylabel("Absolute rotation error (radians)")
plt.title("Rotation absolute error by observation")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "rotation_abs_error_rad.png")
plt.close()

# Optional degrees version
plt.figure(figsize=(12, 5))
plt.bar(idx - bar_width / 2, np.rad2deg(rot_err_bayes), width=bar_width, color=bayes_color, label="Bayesian")
plt.bar(idx + bar_width / 2, np.rad2deg(rot_err_map), width=bar_width, color=map_color, label="MAP")
plt.xlabel("Observation index")
plt.ylabel("Absolute rotation error (degrees)")
plt.title("Rotation absolute error by observation")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "rotation_abs_error_deg.png")
plt.close()

print(f"Saved plots to {plot_dir}")