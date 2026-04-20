import torch
import torch.nn as nn

from src.grid_funcs import GridParams


class BaseModel(nn.Module):
    def __init__(
        self,
        v_params: GridParams,
        K: int,
        beta: float,
        Z_x: torch.Tensor,
        Z_x_inv: torch.Tensor,
    ):
        super().__init__()

        self.grid = v_params
        self.K = K
        self.beta = beta
        self.downsample_ratio = v_params.downsample_ratio

        self.hr_shape = v_params.hr_shape
        self.lr_shape = v_params.lr_shape

        self.N = int(self.hr_shape[0] * self.hr_shape[1])
        self.M = int(self.lr_shape[0] * self.lr_shape[1])

        # Optimize only K-1 relative transforms.
        self.shift_params = nn.Parameter(torch.zeros(max(K - 1, 0), 2))
        self.rot_params = nn.Parameter(torch.zeros(max(K - 1, 0)))

        # Paper initializes gamma at 4.0 for 4x SR.
        self.gamma = nn.Parameter(torch.tensor(4.0))

        self.register_buffer("Z_x", Z_x)
        self.register_buffer("Z_x_inv", Z_x_inv)

        self.register_buffer("v_i", v_params.v_i)
        self.register_buffer("v_j", v_params.v_j)
        self.register_buffer("v_avg", v_params.v_avg)

    @property
    def shifts(self):
        if self.K == 1:
            return torch.zeros(1, 2, device=self.gamma.device, dtype=self.gamma.dtype)
        ref = torch.zeros(1, 2, device=self.shift_params.device, dtype=self.shift_params.dtype)
        return torch.cat([ref, self.shift_params], dim=0)

    @property
    def rots(self):
        if self.K == 1:
            return torch.zeros(1, device=self.gamma.device, dtype=self.gamma.dtype)
        ref = torch.zeros(1, device=self.rot_params.device, dtype=self.rot_params.dtype)
        return torch.cat([ref, self.rot_params], dim=0)

    def forward(self, y_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_HR(self, y_obs: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def set_params(self, shifts, rots, gamma):
        with torch.no_grad():
            shifts = torch.as_tensor(shifts, device=self.gamma.device, dtype=self.gamma.dtype)
            rots = torch.as_tensor(rots, device=self.gamma.device, dtype=self.gamma.dtype)

            if self.K > 1:
                self.shift_params.copy_(shifts[1:].to(self.shift_params))
                self.rot_params.copy_(rots[1:].to(self.rot_params))

            self.gamma.copy_(torch.as_tensor(gamma, device=self.gamma.device, dtype=self.gamma.dtype))