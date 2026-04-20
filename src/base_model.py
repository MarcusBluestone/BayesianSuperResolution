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

        self.shifts = nn.Parameter(torch.zeros(K, 2))
        self.rots = nn.Parameter(torch.zeros(K))
        self.gamma = nn.Parameter(torch.tensor(2.0))

        self.register_buffer("Z_x", Z_x)
        self.register_buffer("Z_x_inv", Z_x_inv)

        self.register_buffer("v_i", v_params.v_i)
        self.register_buffer("v_j", v_params.v_j)
        self.register_buffer("v_avg", v_params.v_avg)

    def forward(self, y_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_HR(self, y_obs: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError
    
    def set_params(self, shifts, rots, gamma):
        with torch.no_grad():
            self.shifts.copy_(shifts.to(self.shifts.device, dtype=self.shifts.dtype))
            self.rots.copy_(rots.to(self.rots.device, dtype=self.rots.dtype))
            self.gamma.copy_(torch.as_tensor(gamma, device=self.gamma.device, dtype=self.gamma.dtype))
