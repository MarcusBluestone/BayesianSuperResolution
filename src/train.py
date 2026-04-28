import json
from pathlib import Path
import copy

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

bayes_full_steps = 300

# patch_lr_bounds = (12, 12, 8, 8)   # (lr_top, lr_left, lr_h, lr_w)
patch_lr_bounds = (4, 4, 20, 20)   # Larger patch?
patch_hr_margin = 6

use_true_init = False   # debugging only
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PATHS
# ============================================================
results_dir = Path("imgs/wider_bayes")
data_dir = results_dir / "data"
bayes_dir = results_dir / "bayes"
map_dir = results_dir / "map"

if results_dir.exists():
    raise ValueError("Choose New Directory")

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


def save_loss_plot(losses, path: Path, title: str, stage_boundaries=None):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)

    if stage_boundaries is not None:
        for i, boundary in enumerate(stage_boundaries):
            plt.axvline(
                x=boundary,
                linestyle="--",
                linewidth=1.5,
                color="gray",
                alpha=0.8,
                label="Stage transition" if i == 0 else None,
            )

    plt.legend()
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


def set_trainable(model, *, shifts: bool, rots: bool, gamma: bool, x: bool | None = None):
    if hasattr(model, "shift_params"):
        model.shift_params.requires_grad_(shifts)
    if hasattr(model, "rot_params"):
        model.rot_params.requires_grad_(rots)
    if hasattr(model, "gamma"):
        model.gamma.requires_grad_(gamma)
    if x is not None and hasattr(model, "x"):
        model.x.requires_grad_(x)


def run_stage(
    model: BaseModel,
    y_obs: torch.Tensor,
    lr: float,
    name: str,
    max_steps: int = 800,
    patience: int = 40,
    min_delta: float = 5,
):
    model = model.to(device)
    y_obs = y_obs.to(device=device, dtype=torch.float32)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    losses = []
    best_loss = float("inf")
    best_step = -1
    best_state = copy.deepcopy(model.state_dict())
    steps_since_improvement = 0

    print(f"Beginning stage: {name}")
    for step in tqdm(range(max_steps), desc=name):
        optimizer.zero_grad()
        loss = model(y_obs)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)

        if best_loss - loss_value > min_delta:
            best_loss = loss_value
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        if steps_since_improvement >= patience:
            print(
                f"Early stopping in {name} at step {step + 1} "
                f"(best step: {best_step + 1}, best loss: {best_loss:.6f})"
            )
            break

    model.load_state_dict(best_state)
    return losses, best_loss, best_step

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

print("Setting Up Variables. Inversion is slow...")
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

# Stage 1: shifts only
set_trainable(bayes_patch, shifts=True, rots=False, gamma=False)
losses_1, best_loss_1, best_step_1 = run_stage(
    model=bayes_patch,
    y_obs=y_obs_patch,
    lr=1e-2,
    name="bayes_patch_stage1_shifts",
)

# Stage 2: shifts + rotations
set_trainable(bayes_patch, shifts=True, rots=True, gamma=False)
losses_2, best_loss_2, best_step_2 = run_stage(
    model=bayes_patch,
    y_obs=y_obs_patch,
    lr=5e-3,
    name="bayes_patch_stage2_shifts_rots",
)

# Stage 3: shifts + rotations + gamma
set_trainable(bayes_patch, shifts=True, rots=True, gamma=True)
losses_3, best_loss_3, best_step_3 = run_stage(
    model=bayes_patch,
    y_obs=y_obs_patch,
    lr=2e-3,
    name="bayes_patch_stage3_all",
    max_steps = 2_000
)
all_bayes_losses = losses_1 + losses_2 + losses_3
bayes_stage_boundaries = [
    len(losses_1),
    len(losses_1) + len(losses_2),
]

save_loss_plot(
    all_bayes_losses,
    bayes_dir / "patch" / "loss" / "loss_plot.png",
    title="bayes_patch_staged",
    stage_boundaries=bayes_stage_boundaries,
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
bayes_full_recon = bayes_full.get_HR(
    y_obs.to(device=device, dtype=torch.float32),
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

set_trainable(map_full, shifts=True, rots=False, gamma=False, x=True)
losses_1, _, _ = run_stage(
    model=map_full,
    y_obs=y_obs,
    lr=1e-2,
    name="map_full_stage1_shifts",
)

set_trainable(map_full, shifts=True, rots=True, gamma=False, x=True)
losses_2, _, _ = run_stage(
    model=map_full,
    y_obs=y_obs,
    lr=5e-3,
    name="map_full_stage2_shifts_rots",
)

set_trainable(map_full, shifts=True, rots=True, gamma=True, x=True)
losses_3, _, _ = run_stage(
    model=map_full,
    y_obs=y_obs,
    lr=2e-3,
    name="map_full_stage3_all",
)

all_map_losses = losses_1 + losses_2 + losses_3
map_stage_boundaries = [
    len(losses_1),
    len(losses_1) + len(losses_2),
]

save_loss_plot(
    all_map_losses,
    map_dir / "full" / "loss" / "loss_plot.png",
    title="map_full_staged",
    stage_boundaries=map_stage_boundaries,
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