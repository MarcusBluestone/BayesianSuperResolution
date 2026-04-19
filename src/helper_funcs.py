import torch
import numpy as np
from PIL import Image

from src.grid_funcs import GridParams, build_grid_params


def generate_Z_x(v_params: GridParams, A: float = 0.04, r: float = 1.0) -> torch.Tensor:
    """
    Generates the covariance matrix Z_x for the HR domain in v_params.
    """
    grid = v_params.v_i.t()  # (N, 2)
    dists_sq = torch.cdist(grid, grid, p=2) ** 2
    Z_x = A * torch.exp(-dists_sq / (r ** 2))
    return Z_x


def create_lrs(
    hr_img: torch.Tensor,
    downsample_ratio: int,
    shift_range: list[float],
    rot_range: list[float],
    gamma: float,
    beta: float,
    K: int,
    save_file=None,
) -> torch.Tensor:
    hr_shape = torch.tensor(hr_img.shape[1:])
    lr_shape = hr_shape // downsample_ratio

    grid = build_grid_params(
        hr_shape=hr_shape,
        downsample_ratio=downsample_ratio,
    )

    shifts = torch.rand(K, 2) * (shift_range[1] - shift_range[0]) + shift_range[0]
    rots = torch.deg2rad(
        torch.rand(K) * (rot_range[1] - rot_range[0]) + rot_range[0]
    )

    W = get_W_matrix(
        shifts=shifts,
        rots=rots,
        gamma=torch.tensor(gamma, dtype=torch.float32),
        v_params=grid,
    )

    eps = torch.randn(W.shape[0], 1, dtype=torch.float32) * np.sqrt(1.0 / beta)

    hr_rasterized = hr_img[0].reshape(-1, 1).to(dtype=torch.float32)
    hr_rasterized = hr_rasterized - 0.5

    y = W @ hr_rasterized + eps
    M = y.shape[0] // K
    y_reshaped = y.reshape(K, M, 1)

    if save_file:
        H_lr, W_lr = int(lr_shape[0]), int(lr_shape[1])
        for k in range(K):
            unflattened_y = y_reshaped[k].reshape(H_lr, W_lr)
            display_ready = np.clip(
                255 * (unflattened_y.detach().cpu().numpy() + 0.5),
                0,
                255,
            ).astype(np.uint8)
            Image.fromarray(display_ready).save(save_file / f"lr_{k}.png")

    return y_reshaped


def get_W_matrix(
    shifts: torch.Tensor,
    rots: torch.Tensor,
    gamma: torch.Tensor | float,
    v_params: GridParams,
) -> torch.Tensor:
    v_i = v_params.v_i
    v_j = v_params.v_j
    v_avg = v_params.v_avg

    if not torch.is_tensor(gamma):
        gamma = torch.tensor(gamma, dtype=v_i.dtype, device=v_i.device)
    else:
        gamma = gamma.to(dtype=v_i.dtype, device=v_i.device)

    shifts = shifts.to(dtype=v_i.dtype, device=v_i.device)
    rots = rots.to(dtype=v_i.dtype, device=v_i.device)

    K = shifts.shape[0]
    weights = []

    for k in range(K):
        theta = rots[k]
        R = torch.stack([
            torch.stack([torch.cos(theta), torch.sin(theta)]),
            torch.stack([-torch.sin(theta), torch.cos(theta)]),
        ])

        u_j = R @ (v_j - v_avg) + v_avg + shifts[k].unsqueeze(1)
        dists_sq = torch.cdist(u_j.t(), v_i.t(), p=2) ** 2

        W_k = torch.exp(-dists_sq / (gamma ** 2))
        W_k = W_k / (W_k.sum(dim=1, keepdim=True) + 1e-12)
        weights.append(W_k)

    return torch.cat(weights, dim=0)

def crop_y_obs_to_patch(y_obs: torch.Tensor, lr_bounds: tuple[int, int, int, int]) -> torch.Tensor:
    """
    Crop full LR observations to the patch specified by lr_bounds.

    Args:
        y_obs: (K, M_full, 1)
        lr_bounds: (top, bottom, left, right)

    Returns:
        y_patch: (K, M_patch, 1)
    """
    K, M_full, C = y_obs.shape
    if C != 1:
        raise ValueError("Expected y_obs to have shape (K, M, 1)")

    lr_top, lr_bottom, lr_left, lr_right = lr_bounds
    patch_h = lr_bottom - lr_top
    patch_w = lr_right - lr_left

    side = int(M_full ** 0.5)
    if side * side != M_full:
        raise ValueError(f"Expected square LR images when reshaped, got M={M_full}")

    y_imgs = y_obs.reshape(K, side, side, 1)
    y_patch = y_imgs[:, lr_top:lr_bottom, lr_left:lr_right, :]
    y_patch = y_patch.reshape(K, patch_h * patch_w, 1)

    return y_patch