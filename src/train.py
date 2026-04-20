import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.base_model import BaseModel
from src.bayes_model import BayesModel
from src.grid_funcs import build_grid_params
from src.helper_funcs import create_lrs, crop_y_obs_to_patch, generate_Z_x
from src.map_model import MapModel


# ============================================================
# CONFIG
# ============================================================
hr_shape = torch.tensor([128, 128])   # (H, W) = paper's 384 x 256
K = 16
beta = 400.0
downsample_ratio = 4
shift_range = [-2, 2]
rot_range = [-4, 4]   # degrees
gamma = 2.0

A = 0.04
r = 1.0

bayes_patch_steps = 300
bayes_full_steps = 300
map_full_steps = 300

patch_lr_bounds = (12, 12, 8, 8)   # (lr_top, lr_left, lr_h, lr_w)
patch_hr_margin = 6

use_true_init = False   # debugging only
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# PATHS
# ============================================================
results_dir = Path("imgs/results")
data_dir = results_dir / "data"
bayes_dir = results_dir / "bayes"
map_dir = results_dir / "map"

if results_dir.exists():
    shutil.rmtree(results_dir)

for path in [
    results_dir,
    data_dir,
    data_dir / "lr",
    bayes_dir,
    bayes_dir / "patch",
    bayes_dir / "patch" / "loss",
    bayes_dir / "full",
    map_dir,
    map_dir / "full",
    map_dir / "full" / "loss",
]:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILS
# ============================================================
def tensor_to_uint8_image(img: torch.Tensor):
    img = ((img.detach().cpu() + 0.5) * 255).clamp(0, 255).byte()

    if img.dim() == 2:
        return img.numpy()
    if img.dim() == 3:
        return img.permute(1, 2, 0).numpy()

    raise ValueError(f"Unexpected image shape: {tuple(img.shape)}")


def save_image(img: torch.Tensor, path: Path):
    Image.fromarray(tensor_to_uint8_image(img)).save(path)


def save_loss_plot(losses, path: Path, title: str):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_params(path: Path, shifts, rots, gamma_value):
    payload = {
        "shifts": torch.as_tensor(shifts).detach().cpu().tolist(),
        "rots": torch.as_tensor(rots).detach().cpu().tolist(),
        "gamma": float(torch.as_tensor(gamma_value).detach().cpu().item()),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def train_model(
    model: BaseModel,
    y_obs: torch.Tensor,
    steps: int,
    loss_plot_path: Path,
    name: str,
):
    model = model.to(device)
    y_obs = y_obs.to(device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    print(f"Beginning training: {name}")
    for _ in tqdm(range(steps), desc=name):
        optimizer.zero_grad()
        loss = model(y_obs)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    save_loss_plot(losses, loss_plot_path, title=name)
    return losses


def build_covariances(v_params):
    Z_x = generate_Z_x(v_params, A=A, r=r)
    Z_x_inv = torch.linalg.inv(Z_x)
    return Z_x, Z_x_inv


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
            "K": K,
            "beta": beta,
            "downsample_ratio": downsample_ratio,
            "shift_range": shift_range,
            "rot_range_deg": rot_range,
            "shifts": true_shifts.tolist(),
            "rots": true_rots.tolist(),
            "gamma": gamma,
            "A": A,
            "r": r,
        },
        f,
        indent=2,
    )


# ============================================================
# GRID SETUP
# ============================================================
v_params_patch = build_grid_params(
    hr_shape=hr_shape,
    downsample_ratio=downsample_ratio,
    lr_patch=patch_lr_bounds,
    hr_margin=patch_hr_margin,
)
Z_x_patch, Z_x_patch_inv = build_covariances(v_params_patch)
y_obs_patch = crop_y_obs_to_patch(y_obs, v_params_patch.lr_bounds)

v_params_full = build_grid_params(
    hr_shape=hr_shape,
    downsample_ratio=downsample_ratio,
    lr_patch=None,
    hr_margin=0,
)
Z_x_full, Z_x_full_inv = build_covariances(v_params_full)


# ============================================================
# BAYESIAN: PATCH TRAINING -> FULL RECONSTRUCTION
# ============================================================
print("\n" + "=" * 60)
print("BAYESIAN PIPELINE")
print("=" * 60)

bayes_patch = BayesModel(
    v_params=v_params_patch,
    K=K,
    beta=beta,
    Z_x=Z_x_patch,
    Z_x_inv=Z_x_patch_inv,
)

if use_true_init:
    bayes_patch.set_params(true_shifts, true_rots, gamma)

train_model(
    model=bayes_patch,
    y_obs=y_obs_patch,
    steps=bayes_patch_steps,
    loss_plot_path=bayes_dir / "patch" / "loss" / "loss_plot.png",
    name="bayes_patch",
)

bayes_learned_shifts = bayes_patch.shifts.detach().cpu()
bayes_learned_rots = bayes_patch.rots.detach().cpu()
bayes_learned_gamma = bayes_patch.gamma.detach().cpu()

save_params(
    bayes_dir / "patch" / "learned_params.json",
    bayes_learned_shifts,
    bayes_learned_rots,
    bayes_learned_gamma,
)

bayes_full = BayesModel(
    v_params=v_params_full,
    K=K,
    beta=beta,
    Z_x=Z_x_full,
    Z_x_inv=Z_x_full_inv,
)
bayes_full.set_params(bayes_learned_shifts, bayes_learned_rots, bayes_learned_gamma)
bayes_full = bayes_full.to(device)

print("Running Bayesian full reconstruction...")
bayes_full_recon = bayes_full.reconstruct_full_iterative(
    y_obs.to(device=device, dtype=torch.float32),
    steps=bayes_full_steps,
    lr=1e-2,
)

save_image(bayes_full_recon, bayes_dir / "full" / "reconstruction.png")
save_params(
    bayes_dir / "full" / "params_used.json",
    bayes_learned_shifts,
    bayes_learned_rots,
    bayes_learned_gamma,
)


# ============================================================
# MAP: FULL TRAINING ONLY
# ============================================================
print("\n" + "=" * 60)
print("MAP PIPELINE (FULL ONLY)")
print("=" * 60)

map_full = MapModel(
    v_params=v_params_full,
    K=K,
    beta=beta,
    Z_x=Z_x_full,
    Z_x_inv=Z_x_full_inv,
)

if use_true_init:
    map_full.set_params(true_shifts, true_rots, gamma)

train_model(
    model=map_full,
    y_obs=y_obs,
    steps=map_full_steps,
    loss_plot_path=map_dir / "full" / "loss" / "loss_plot.png",
    name="map_full",
)

map_learned_shifts = map_full.shifts.detach().cpu()
map_learned_rots = map_full.rots.detach().cpu()
map_learned_gamma = map_full.gamma.detach().cpu()

save_params(
    map_dir / "full" / "learned_params.json",
    map_learned_shifts,
    map_learned_rots,
    map_learned_gamma,
)
save_image(map_full.get_HR(), map_dir / "full" / "reconstruction.png")

print("\nDone.")
print(f"Results saved to: {results_dir}")