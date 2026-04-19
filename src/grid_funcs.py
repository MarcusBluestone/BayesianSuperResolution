import torch
from dataclasses import dataclass


@dataclass
class GridParams:
    # HR pixel coordinates, shape (2, N)
    v_i: torch.Tensor

    # LR pixel coordinates expressed in HR units, shape (2, M)
    v_j: torch.Tensor

    # Center used for rotation, shape (2, 1)
    v_avg: torch.Tensor

    # Bounds: (top, bottom, left, right)
    hr_bounds: tuple[int, int, int, int]
    lr_bounds: tuple[int, int, int, int]

    # Local shapes for the current model domain
    hr_shape: torch.Tensor
    lr_shape: torch.Tensor

    downsample_ratio: int


def get_grid_from_ranges(row_start, row_end, col_start, col_end, scale=1.0):
    rows = torch.arange(row_start, row_end)
    cols = torch.arange(col_start, col_end)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    grid = torch.stack([yy.flatten(), xx.flatten()], dim=0).float()

    if scale != 1.0:
        grid = grid * scale + (scale - 1) / 2.0

    return grid


def build_grid_params(
    hr_shape,
    downsample_ratio,
    lr_patch=None,
    hr_margin=0,
) -> GridParams:
    """
    Returns grid metadata for either the full image or a patch.

    Args:
        hr_shape: (H_hr, W_hr)
        downsample_ratio: integer
        lr_patch: None or (lr_top, lr_left, lr_h, lr_w)
        hr_margin: HR padding around mapped LR patch
    """
    if not torch.is_tensor(hr_shape):
        hr_shape = torch.tensor(hr_shape)

    H_hr, W_hr = int(hr_shape[0]), int(hr_shape[1])
    lr_shape_full = hr_shape // downsample_ratio
    H_lr, W_lr = int(lr_shape_full[0]), int(lr_shape_full[1])

    if lr_patch is None:
        hr_top, hr_bottom, hr_left, hr_right = 0, H_hr, 0, W_hr
        lr_top, lr_bottom, lr_left, lr_right = 0, H_lr, 0, W_lr
    else:
        lr_top, lr_left, lr_h, lr_w = lr_patch
        lr_bottom = lr_top + lr_h
        lr_right = lr_left + lr_w

        if not (0 <= lr_top < lr_bottom <= H_lr and 0 <= lr_left < lr_right <= W_lr):
            raise ValueError("lr_patch is out of LR image bounds")

        hr_top = max(0, lr_top * downsample_ratio - hr_margin)
        hr_left = max(0, lr_left * downsample_ratio - hr_margin)
        hr_bottom = min(H_hr, lr_bottom * downsample_ratio + hr_margin)
        hr_right = min(W_hr, lr_right * downsample_ratio + hr_margin)

    v_i = get_grid_from_ranges(hr_top, hr_bottom, hr_left, hr_right, scale=1.0)
    v_j = get_grid_from_ranges(lr_top, lr_bottom, lr_left, lr_right, scale=downsample_ratio)
    v_avg = v_i.mean(dim=1, keepdim=True)

    return GridParams(
        v_i=v_i,
        v_j=v_j,
        v_avg=v_avg,
        hr_bounds=(hr_top, hr_bottom, hr_left, hr_right),
        lr_bounds=(lr_top, lr_bottom, lr_left, lr_right),
        hr_shape=torch.tensor([hr_bottom - hr_top, hr_right - hr_left]),
        lr_shape=torch.tensor([lr_bottom - lr_top, lr_right - lr_left]),
        downsample_ratio=downsample_ratio,
    )