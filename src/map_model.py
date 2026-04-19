import torch
import torch.nn as nn

from src.helper_funcs import get_W_matrix
from src.base_model import BaseModel


class MapModel(BaseModel):
    def __init__(self, *params, **kw_params):
        super().__init__(*params, **kw_params)
        self.x = nn.Parameter(torch.zeros(self.N, 1))

    def forward(self, y_obs: torch.Tensor):
        K, M, _ = y_obs.shape

        W = get_W_matrix(
            self.shifts,
            self.rots,
            self.gamma,
            self.grid,
        )
        y_flat = y_obs.reshape(K * M, 1)

        y_pred = W @ self.x
        likelihood_loss = torch.sum((y_pred - y_flat) ** 2)
        prior_loss = 0.5 * (self.x.t() @ self.Z_x_inv @ self.x)

        return likelihood_loss + prior_loss

    def get_HR(self, y_obs: torch.Tensor | None = None) -> torch.Tensor:
        return self.x.view(*self.hr_shape.tolist()).detach().cpu()