import copy
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.base_model import BaseModel
from src.helper_funcs import generate_Z_x


# ============================================================
# IMAGE I/O
# ============================================================

def tensor_to_uint8_image(img: torch.Tensor) -> np.ndarray:
    img = ((img.detach().cpu() + 0.5) * 255).clamp(0, 255).byte()
    if img.dim() == 2:
        return img.numpy()
    if img.dim() == 3:
        return img.permute(1, 2, 0).numpy()
    raise ValueError(f"Unexpected image shape: {tuple(img.shape)}")


def save_image(img: torch.Tensor, path: Path) -> None:
    Image.fromarray(tensor_to_uint8_image(img)).save(path)


# ============================================================
# PLOTTING
# ============================================================

def save_loss_plot(
    losses: list[float],
    path: Path,
    title: str,
    stage_boundaries: list[int] | None = None,
) -> None:
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


# ============================================================
# PARAM SERIALISATION
# ============================================================

def save_params(path: Path, shifts, rots, gamma_value) -> None:
    payload = {
        "shifts": torch.as_tensor(shifts).detach().cpu().tolist(),
        "rots": torch.as_tensor(rots).detach().cpu().tolist(),
        "gamma": float(torch.as_tensor(gamma_value).detach().cpu().item()),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# TRAINING HELPERS
# ============================================================

def set_trainable(
    model,
    *,
    shifts: bool,
    rots: bool,
    gamma: bool,
    x: bool | None = None,
) -> None:
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
    device: str,
    max_steps: int = 800,
    patience: int = 40,
    min_delta: float = 5,
) -> tuple[list[float], float, int]:
    """
    Run one optimisation stage with LBFGS + early stopping.

    Returns (losses, best_loss, best_step).
    """

    model = model.to(device)
    y_obs = y_obs.to(device=device, dtype=torch.float32)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.LBFGS(
        params, max_iter=20, lr=lr, line_search_fn="strong_wolfe"
    )

    losses: list[float] = []
    best_loss = float("inf")
    best_step = -1
    best_state = copy.deepcopy(model.state_dict())
    steps_since_improvement = 0

    print(f"Beginning stage: {name}")
    for step in tqdm(range(max_steps), desc=name):
        def closure():
            optimizer.zero_grad()
            loss = model(y_obs)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            loss_value = float(model(y_obs).item())

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


# ============================================================
# COVARIANCE BUILDER
# ============================================================

def build_covariances(
    v_params,
    A: float,
    r: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    Z_x = generate_Z_x(v_params, A=A, r=r)
    Z_x_inv = torch.linalg.inv(Z_x)
    return Z_x, Z_x_inv


# ============================================================
# STAGED TRAINING (reusable for any model type)
# ============================================================

def run_three_stage_training(
    model: BaseModel,
    y_obs: torch.Tensor,
    device: str,
    name_prefix: str,
    has_x: bool = False,
    stage3_max_steps: int = 800,
) -> tuple[list[float], list[int]]:
    """
    Runs the standard 3-stage curriculum:
      Stage 1 – shifts only
      Stage 2 – shifts + rotations
      Stage 3 – shifts + rotations + gamma  (+ x if has_x)

    Returns (all_losses, stage_boundaries).
    """
    set_trainable(model, shifts=True, rots=False, gamma=False,
                  x=True if has_x else None)
    losses_1, _, _ = run_stage(
        model=model, y_obs=y_obs, lr=1e-2,
        name=f"{name_prefix}_stage1_shifts", device=device,
    )

    set_trainable(model, shifts=True, rots=True, gamma=False,
                  x=True if has_x else None)
    losses_2, _, _ = run_stage(
        model=model, y_obs=y_obs, lr=5e-3,
        name=f"{name_prefix}_stage2_shifts_rots", device=device,
    )

    set_trainable(model, shifts=True, rots=True, gamma=True,
                  x=True if has_x else None)
    losses_3, _, _ = run_stage(
        model=model, y_obs=y_obs, lr=2e-3,
        name=f"{name_prefix}_stage3_all", device=device,
        max_steps=stage3_max_steps,
    )

    all_losses = losses_1 + losses_2 + losses_3
    boundaries = [len(losses_1), len(losses_1) + len(losses_2)]
    return all_losses, boundaries