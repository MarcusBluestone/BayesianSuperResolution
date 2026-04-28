import torch

from src.helper_funcs import get_W_matrix
from src.base_model import BaseModel


class BayesModel(BaseModel):
    """
    Bayesian super-resolution model following Tipping & Bishop.

    This class optimizes the marginal likelihood obtained by integrating out
    the high-resolution image x
    """

    def __init__(self, *params, **kw_params):
        super().__init__(*params, **kw_params)

    def _posterior_mu_and_Sigma_inv(self, y_obs: torch.Tensor):
        """
        Compute posterior quantities from Equations (11) and (12).
        """
        device = self.Z_x.device
        dtype = self.Z_x.dtype

        y_obs = y_obs.to(device=device, dtype=dtype)
        K, M, _ = y_obs.shape

        # Accumulate beta sum_k W_k^T W_k
        WtW_sum = torch.zeros(self.N, self.N, device=device, dtype=dtype)

        # Accumulate beta sum_k W_k^T y_k
        Wty_sum = torch.zeros(self.N, 1, device=device, dtype=dtype)

        W_list = []

        for k in range(K):
            y_k = y_obs[k]  # shape: (M, 1)

            W_k = get_W_matrix(
                self.shifts[k:k + 1],
                self.rots[k:k + 1],
                self.gamma,
                self.grid,
            )  # shape: (M, N)

            W_list.append(W_k)

            WtW_sum = WtW_sum + self.beta * (W_k.T @ W_k)
            Wty_sum = Wty_sum + self.beta * (W_k.T @ y_k)

        # Eq. (11): Sigma^{-1}
        Sigma_inv = self.Z_x_inv + WtW_sum

        sign_Sigma_inv, logdet_Sigma_inv = torch.linalg.slogdet(Sigma_inv)
        if sign_Sigma_inv <= 0:
            raise ValueError("Sigma^{-1} is not positive definite")

        # Eq. (12): mu = Sigma * beta sum_k W_k^T y_k
        mu = torch.linalg.solve(Sigma_inv, Wty_sum)

        return mu, Sigma_inv, logdet_Sigma_inv, W_list, y_obs

    def forward(self, y_obs: torch.Tensor):
        """
        Return the negative marginal log likelihood, dropping constants.
        """
        mu, Sigma_inv, logdet_Sigma_inv, W_list, y_obs = (
            self._posterior_mu_and_Sigma_inv(y_obs)
        )

        device = self.Z_x.device
        dtype = self.Z_x.dtype
        K, _, _ = y_obs.shape

        # Eq. (15): beta sum_k ||y_k - W_k mu||^2
        recon_error = torch.tensor(0.0, device=device, dtype=dtype)

        for k in range(K):
            residual = y_obs[k] - W_list[k] @ mu
            recon_error = recon_error + self.beta * (residual.T @ residual).squeeze()

        # Eq. (15): mu^T Z_x^{-1} mu
        prior_term = (mu.T @ self.Z_x_inv @ mu).squeeze()

        # Eq. (15): -log|Sigma| = log|Sigma^{-1}|
        logdet_term = logdet_Sigma_inv

        neg_log_likelihood = 0.5 * (
            recon_error
            + prior_term
            + logdet_term
        )

        return neg_log_likelihood

    def get_HR(self, y_obs: torch.Tensor | None = None):
        """
        Return the posterior mean high-resolution image.
        """
        with torch.no_grad():
            if y_obs is None:
                raise ValueError("BayesModel.get_HR requires y_obs")

            mu, _, _, _, _ = self._posterior_mu_and_Sigma_inv(y_obs)

            return mu.view(*self.hr_shape.tolist()).detach().cpu()