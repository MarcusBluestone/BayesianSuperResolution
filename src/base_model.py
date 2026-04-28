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

        self.K = K
        self.downsample_ratio = v_params.downsample_ratio

        self.hr_shape = v_params.hr_shape
        self.lr_shape = v_params.lr_shape

        self.N = int(self.hr_shape[0] * self.hr_shape[1])
        self.M = int(self.lr_shape[0] * self.lr_shape[1])

        # Optimize only K - 1 relative transforms.
        # Image 0 is the fixed reference transform.
        self.shift_params = nn.Parameter(torch.zeros(max(K - 1, 0), 2))
        self.rot_params = nn.Parameter(torch.zeros(max(K - 1, 0)))

        # Learnable blur / PSF width.
        self.gamma = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))

        # Model buffers. These move automatically with model.to(device).
        self.register_buffer("Z_x", Z_x)
        self.register_buffer("Z_x_inv", Z_x_inv)
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))

        # Grid tensors as buffers. Do NOT store a persistent GridParams object,
        # because it can keep stale CPU tensor references after model.to("cuda").
        self.hr_bounds = v_params.hr_bounds
        self.lr_bounds = v_params.lr_bounds
        self.register_buffer("grid_v_i", v_params.v_i)
        self.register_buffer("grid_v_j", v_params.v_j)
        self.register_buffer("grid_v_avg", v_params.v_avg)

    @property
    def grid(self) -> GridParams:
        return GridParams(
            v_i=self.grid_v_i,
            v_j=self.grid_v_j,
            v_avg=self.grid_v_avg,
            hr_bounds=self.hr_bounds,
            lr_bounds=self.lr_bounds,
            hr_shape=self.hr_shape,
            lr_shape=self.lr_shape,
            downsample_ratio=self.downsample_ratio,
        )
    @property
    def shifts(self) -> torch.Tensor:
        """
        Full K shifts, with image 0 fixed at zero and images 1:K optimized.
        Gradients flow into self.shift_params.
        """
        ref = torch.zeros(
            1,
            2,
            device=self.shift_params.device,
            dtype=self.shift_params.dtype,
        )

        if self.K == 1:
            return ref

        return torch.cat([ref, self.shift_params], dim=0)

    @property
    def rots(self) -> torch.Tensor:
        """
        Full K rotations, with image 0 fixed at zero and images 1:K optimized.
        Gradients flow into self.rot_params.
        """
        ref = torch.zeros(
            1,
            device=self.rot_params.device,
            dtype=self.rot_params.dtype,
        )

        if self.K == 1:
            return ref

        return torch.cat([ref, self.rot_params], dim=0)

    def forward(self, y_obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_HR(self, y_obs: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def set_params(self, shifts, rots, gamma):
        """
        Set initial transform/gamma values.

        shifts: shape (K, 2), where shifts[0] should usually be [0, 0]
        rots: shape (K,), where rots[0] should usually be 0
        gamma: scalar
        """
        with torch.no_grad():
            shifts = torch.as_tensor(
                shifts,
                device=self.shift_params.device,
                dtype=self.shift_params.dtype,
            )
            rots = torch.as_tensor(
                rots,
                device=self.rot_params.device,
                dtype=self.rot_params.dtype,
            )

            if self.K > 1:
                self.shift_params.copy_(shifts[1:])
                self.rot_params.copy_(rots[1:])

            self.gamma.copy_(
                torch.as_tensor(
                    gamma,
                    device=self.gamma.device,
                    dtype=self.gamma.dtype,
                )
            )