import torch
from tqdm import tqdm

from src.helper_funcs import get_W_matrix
from src.base_model import BaseModel


class BayesModel(BaseModel):
    def __init__(self, *params, **kw_params):
        super().__init__(*params, **kw_params)

    def forward(self, y_obs: torch.Tensor):
        """
        Negative marginal log likelihood based on Eq. 15.
        """
        K, M, _ = y_obs.shape
        device = self.Z_x.device
        dtype = self.Z_x.dtype

        info_matrix = torch.zeros(self.N, self.N, device=device, dtype=dtype)
        info_vector = torch.zeros(self.N, 1, device=device, dtype=dtype)
        W_list = []

        for k in range(K):
            y_k = y_obs[k]  # (M, 1)

            W_k = get_W_matrix(
                self.shifts[k:k+1],
                self.rots[k:k+1],
                self.gamma,
                self.grid,
            )  # (M, N)

            W_list.append(W_k)
            info_matrix = info_matrix + self.beta * (W_k.t() @ W_k)
            info_vector = info_vector + self.beta * (W_k.t() @ y_k)

        # Eq. 11
        post_cov_inv = self.Z_x_inv + info_matrix
        post_cov = torch.linalg.inv(post_cov_inv)

        # Eq. 12
        post_mean = post_cov @ info_vector

        # Eq. 15 reconstruction term
        recon_error = torch.tensor(0.0, device=device, dtype=dtype)
        for k in range(K):
            y_k = y_obs[k]
            W_k = W_list[k]
            residual = y_k - W_k @ post_mean
            recon_error = recon_error + self.beta * (residual.t() @ residual).squeeze()

        # Eq. 15 prior term
        prior_term = (post_mean.t() @ self.Z_x_inv @ post_mean).squeeze()

        # Eq. 15 log-det terms
        sign_Zx, logdet_Zx = torch.linalg.slogdet(self.Z_x)
        sign_post_cov_inv, logdet_post_cov_inv = torch.linalg.slogdet(post_cov_inv)

        if sign_Zx <= 0:
            raise ValueError("Z_x is not positive definite")
        if sign_post_cov_inv <= 0:
            raise ValueError("Posterior precision is not positive definite")

        logdet_post_cov = -logdet_post_cov_inv

        beta_t = torch.as_tensor(self.beta, device=device, dtype=dtype)

        neg_log_likelihood = 0.5 * (
            recon_error
            + prior_term
            + logdet_Zx
            - logdet_post_cov
            - K * M * torch.log(beta_t)
        )

        return neg_log_likelihood
    
    def get_HR(self, y_obs: torch.Tensor | None = None):
        """
        Return posterior mean of HR image (Eq. 11-12 from Tipping & Bishop 2003)
        
        This is the maximum a posteriori (MAP) estimate, which for Gaussian posteriors
        equals the posterior mean.
        """
        if y_obs is None:
            raise ValueError("BayesModel.get_HR requires y_obs")

        K, M, _ = y_obs.shape

        # === Compute JOINT posterior (Eq. 11-12) ===
        
        # Accumulate information from ALL K observations
        info_matrix = torch.zeros(self.N, self.N, device=self.Z_x.device, dtype=self.Z_x.dtype)
        info_vector = torch.zeros(self.N, 1, device=self.Z_x.device, dtype=self.Z_x.dtype)
        
        for k in range(K):
            y_k = y_obs[k]  # (M, 1)
            
            W_k = get_W_matrix(
                self.shifts[k:k+1],
                self.rots[k:k+1],
                self.gamma,
                self.grid,
            )  # (M, N)

            # Accumulate: β * Σ_k W_k^T W_k
            info_matrix = info_matrix + self.beta * W_k.t() @ W_k
            # Accumulate: β * Σ_k W_k^T y_k
            info_vector = info_vector + self.beta * W_k.t() @ y_k

        # Posterior covariance: Σ = [Z_x^-1 + β * Σ_k W_k^T W_k]^-1 (Eq. 11)
        post_cov_inv = self.Z_x_inv + info_matrix
        post_cov = torch.linalg.inv(post_cov_inv)
        
        # Posterior mean: μ = β * Σ * Σ_k W_k^T y_k (Eq. 12)
        # Note: info_vector already includes beta factor, so we just do post_cov @ info_vector
        x_post_mean = post_cov @ info_vector

        return x_post_mean.view(*self.hr_shape.tolist()).detach().cpu()
    
    def reconstruct_full_iterative(
        self,
        y_obs: torch.Tensor,
        steps: int = 300,
        lr: float = 1e-2,
        x_init: torch.Tensor | None = None,
    ):
        """
        Reconstruct full HR image by maximizing the numerator of Eq. (9),
        matching the paper's full-image procedure.
        """
        if y_obs is None:
            raise ValueError("reconstruct_full_iterative requires y_obs")

        device = self.Z_x.device
        dtype = self.Z_x.dtype
        K, M, _ = y_obs.shape

        y_obs = y_obs.to(device=device, dtype=dtype)

        # initialize x
        if x_init is None:
            x = torch.zeros(self.N, 1, device=device, dtype=dtype, requires_grad=True)
        else:
            x = x_init.to(device=device, dtype=dtype).reshape(self.N, 1).clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in tqdm(list(range(steps)), "Iterating for HR"):
            optimizer.zero_grad()

            recon_term = torch.tensor(0.0, device=device, dtype=dtype)
            for k in range(K):
                y_k = y_obs[k]
                W_k = get_W_matrix(
                    self.shifts[k:k+1].detach(),
                    self.rots[k:k+1].detach(),
                    self.gamma.detach(),
                    self.grid,
                )
                residual = y_k - W_k @ x
                recon_term = recon_term + self.beta * (residual.t() @ residual).squeeze()

            prior_term = (x.t() @ self.Z_x_inv @ x).squeeze()

            loss = 0.5 * (recon_term + prior_term)
            loss.backward()
            optimizer.step()

        return x.detach().view(*self.hr_shape.tolist())