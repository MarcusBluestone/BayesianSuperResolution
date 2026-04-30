import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.bayes_model import BayesModel
from src.grid_funcs import build_grid_params
from src.helper_funcs import create_lrs, crop_y_obs_to_patch
from src.map_model import MapModel
from src.train_utils import (
    build_covariances,
    run_three_stage_training,
    save_image,
    save_loss_plot,
    save_params,
    set_trainable,
    run_stage,
)


# ============================================================
# CONFIG
# ============================================================
hr_shape = torch.tensor([128, 128])
K = 16
beta = 400.0
downsample_ratio = 4
shift_range = [-2, 2]
rot_range = [-4, 4]       # degrees
gamma = 2.0

A = 0.04
r = 1.0

patch_lr_bounds = (11, 11, 9, 9)   # (lr_top, lr_left, lr_h, lr_w) — centred 9x9 patch
patch_lr_bounds = (4, 4, 20, 20)   # (lr_top, lr_left, lr_h, lr_w) — centred 9x9 patch

patch_hr_margin = 5

use_true_init = False
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DIRECTORY LAYOUT
# ============================================================
results_dir = Path("imgs/bigger_patch")
data_dir    = results_dir / "data"
bayes_dir   = results_dir / "bayes"
map_full_dir   = results_dir / "map_full"
map_patch_dir  = results_dir / "map_patch"

if results_dir.exists():
    raise ValueError("Choose a new results directory")

for path in [
    results_dir,
    data_dir, data_dir / "lr",
    bayes_dir, bayes_dir / "patch", bayes_dir / "patch" / "loss", bayes_dir / "full",
    map_full_dir, map_full_dir / "full", map_full_dir / "full" / "loss",
    map_patch_dir, map_patch_dir / "patch", map_patch_dir / "patch" / "loss",
    map_patch_dir, map_patch_dir / "full",
]:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA GENERATION
# ============================================================
print("Loading HR image...")
hr_img = Image.open("imgs/hr.jpg").convert("L")
hr_img = hr_img.resize(hr_shape.tolist()[::-1])   # PIL expects (W, H)
hr_img.save(data_dir / "hr_resized.png")
hr_img = transforms.ToTensor()(hr_img)

print(f"Creating {K} LR observations...")
y_obs, true_shifts, true_rots = create_lrs(
    hr_img=hr_img,
    downsample_ratio=downsample_ratio,
    shift_range=shift_range,
    rot_range=rot_range,
    gamma=gamma,
    beta=beta,
    K=K,
    save_file=data_dir / "lr",
)

with open(data_dir / "true_values.json", "w") as f:
    json.dump(
        {
            "hr_shape_hw": hr_shape.tolist(),
            "K": K, "beta": beta,
            "downsample_ratio": downsample_ratio,
            "shift_range": shift_range,
            "rot_range_deg": rot_range,
            "shifts": true_shifts.tolist(),
            "rots": true_rots.tolist(),
            "gamma": gamma,
            "A": A, "r": r,
        },
        f, indent=2,
    )


# ============================================================
# GRID + COVARIANCE SETUP
# ============================================================
print("Setting up grids and covariances (inversion is slow)...")

v_params_patch = build_grid_params(
    hr_shape=hr_shape,
    downsample_ratio=downsample_ratio,
    lr_patch=patch_lr_bounds,
    hr_margin=patch_hr_margin,
)
Z_x_patch, Z_x_patch_inv = build_covariances(v_params_patch, A=A, r=r)
y_obs_patch = crop_y_obs_to_patch(y_obs, v_params_patch.lr_bounds)

v_params_full = build_grid_params(
    hr_shape=hr_shape,
    downsample_ratio=downsample_ratio,
    lr_patch=None,
    hr_margin=0,
)
Z_x_full, Z_x_full_inv = build_covariances(v_params_full, A=A, r=r)


# ============================================================
# PIPELINE 1: BAYESIAN  (patch param-estimation -> full recon)
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE 1: BAYESIAN")
print("=" * 60)

bayes_patch = BayesModel(
    v_params=v_params_patch, K=K, beta=beta,
    Z_x=Z_x_patch, Z_x_inv=Z_x_patch_inv,
)
if use_true_init:
    bayes_patch.set_params(true_shifts, true_rots, gamma)

all_bayes_losses, bayes_boundaries = run_three_stage_training(
    model=bayes_patch,
    y_obs=y_obs_patch,
    device=device,
    name_prefix="bayes_patch",
    has_x=False,
    stage3_max_steps=2_000,
)

save_loss_plot(
    all_bayes_losses,
    bayes_dir / "patch" / "loss" / "loss_plot.png",
    title="bayes_patch_staged",
    stage_boundaries=bayes_boundaries,
)

bayes_shifts = bayes_patch.shifts.detach().cpu()
bayes_rots   = bayes_patch.rots.detach().cpu()
bayes_gamma  = bayes_patch.gamma.detach().cpu()
save_params(bayes_dir / "patch" / "learned_params.json", bayes_shifts, bayes_rots, bayes_gamma)

# Full reconstruction with learned params
bayes_full = BayesModel(
    v_params=v_params_full, K=K, beta=beta,
    Z_x=Z_x_full, Z_x_inv=Z_x_full_inv,
)
bayes_full.set_params(bayes_shifts, bayes_rots, bayes_gamma)
bayes_full = bayes_full.to(device)

print("Running Bayesian full reconstruction...")
bayes_recon = bayes_full.get_HR(y_obs.to(device=device, dtype=torch.float32))
save_image(bayes_recon, bayes_dir / "full" / "reconstruction.png")
save_params(bayes_dir / "full" / "params_used.json", bayes_shifts, bayes_rots, bayes_gamma)


# ============================================================
# PIPELINE 2: MAP — FULL IMAGE
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE 2: MAP (full image)")
print("=" * 60)

map_full_model = MapModel(
    v_params=v_params_full, K=K, beta=beta,
    Z_x=Z_x_full, Z_x_inv=Z_x_full_inv,
)
if use_true_init:
    map_full_model.set_params(true_shifts, true_rots, gamma)

all_map_full_losses, map_full_boundaries = run_three_stage_training(
    model=map_full_model,
    y_obs=y_obs,
    device=device,
    name_prefix="map_full",
    has_x=True,
)

save_loss_plot(
    all_map_full_losses,
    map_full_dir / "full" / "loss" / "loss_plot.png",
    title="map_full_staged",
    stage_boundaries=map_full_boundaries,
)

map_full_shifts = map_full_model.shifts.detach().cpu()
map_full_rots   = map_full_model.rots.detach().cpu()
map_full_gamma  = map_full_model.gamma.detach().cpu()
save_params(map_full_dir / "full" / "learned_params.json", map_full_shifts, map_full_rots, map_full_gamma)
save_image(map_full_model.get_HR(), map_full_dir / "full" / "reconstruction.png")


# ============================================================
# PIPELINE 3: MAP — PATCH  (apples-to-apples with Bayesian)
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE 3: MAP (patch, same patch as Bayesian)")
print("=" * 60)

# The MAP patch model needs a v_params that covers only the patch HR region,
# AND its x parameter only spans those HR pixels.  We reuse v_params_patch
# directly — MapModel stores x with shape N = hr_patch_h * hr_patch_w.
map_patch_model = MapModel(
    v_params=v_params_patch, K=K, beta=beta,
    Z_x=Z_x_patch, Z_x_inv=Z_x_patch_inv,
)
if use_true_init:
    map_patch_model.set_params(true_shifts, true_rots, gamma)

# Train on the SAME cropped y_obs_patch that Bayesian uses.
all_map_patch_losses, map_patch_boundaries = run_three_stage_training(
    model=map_patch_model,
    y_obs=y_obs_patch,
    device=device,
    name_prefix="map_patch",
    has_x=True,
)

save_loss_plot(
    all_map_patch_losses,
    map_patch_dir / "patch" / "loss" / "loss_plot.png",
    title="map_patch_staged",
    stage_boundaries=map_patch_boundaries,
)

map_patch_shifts = map_patch_model.shifts.detach().cpu()
map_patch_rots   = map_patch_model.rots.detach().cpu()
map_patch_gamma  = map_patch_model.gamma.detach().cpu()
save_params(map_patch_dir / "patch" / "learned_params.json", map_patch_shifts, map_patch_rots, map_patch_gamma)

# Reconstruct the full HR image using the learned patch params in a fresh full model.
# (MAP patch only optimised the patch HR pixels; for a fair visual we need a full recon.)
map_patch_full_recon_model = MapModel(
    v_params=v_params_full, K=K, beta=beta,
    Z_x=Z_x_full, Z_x_inv=Z_x_full_inv,
)
map_patch_full_recon_model.set_params(map_patch_shifts, map_patch_rots, map_patch_gamma)
map_patch_full_recon_model = map_patch_full_recon_model.to(device)

# Fix shifts/rots/gamma and only optimise x for a clean reconstruction.
set_trainable(map_patch_full_recon_model, shifts=False, rots=False, gamma=False, x=True)
run_stage(
    model=map_patch_full_recon_model,
    y_obs=y_obs.to(device=device, dtype=torch.float32),
    lr=1e-2,
    name="map_patch_full_recon_x_only",
    device=device,
    max_steps=400,
)

save_image(map_patch_full_recon_model.get_HR(), map_patch_dir / "full" / "reconstruction.png")
save_params(map_patch_dir / "full" / "params_used.json", map_patch_shifts, map_patch_rots, map_patch_gamma)


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Results saved to: {results_dir}")
print()
print(f"{'Model':<20} {'gamma':>8}  {'mean |shift err|':>18}  {'mean |rot err| (deg)':>22}")
print("-" * 74)

import numpy as np

def summarise(label, learned_shifts, learned_rots, learned_gamma):
    s_err = float(torch.mean(torch.norm(learned_shifts - true_shifts, dim=1)).item())
    r_err = float(torch.mean(torch.abs(learned_rots - true_rots)).item())
    r_err_deg = float(np.degrees(r_err))
    g = float(learned_gamma.item())
    print(f"{label:<20} {g:>8.3f}  {s_err:>18.4f}  {r_err_deg:>22.4f}")

summarise("Bayes (patch)",   bayes_shifts,     bayes_rots,     bayes_gamma)
summarise("MAP (full)",      map_full_shifts,  map_full_rots,  map_full_gamma)
summarise("MAP (patch)",     map_patch_shifts, map_patch_rots, map_patch_gamma)
print(f"{'True gamma':<20} {gamma:>8.3f}")