import torch

from src.helper_funcs import get_W_matrix
from src.base_model import BaseModel


class BayesModel(BaseModel):
    def __init__(self, *params, **kw_params):
        super().__init__(*params, **kw_params)

    def forward(self, y_obs: torch.Tensor):
        """
        Compute marginal likelihood of observations (Eq. 15 from Tipping & Bishop 2003)
        
        log p(y|{s_k, θ_k}, γ) = -1/2 [
            β * Σ_k ||y^(k) - W^(k)μ||² + μ^T Z_x^-1 μ + 
            log|Z_x| - log|Σ| - K*M*log(β)
        ]
        """
        K, M, _ = y_obs.shape

        # === Step 1: Accumulate information from ALL K observations ===
        # This computes Σ = [Z_x^-1 + β * Σ_k W_k^T W_k]^-1 (Eq. 11)
        
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
            
            # Accumulate information: β * Σ_k W_k^T W_k
            info_matrix = info_matrix + self.beta * W_k.t() @ W_k
            # Accumulate information vector: β * Σ_k W_k^T y_k
            info_vector = info_vector + self.beta * W_k.t() @ y_k
        
        # === Step 2: Compute JOINT posterior covariance and mean ===
        # Σ = [Z_x^-1 + β * Σ_k W_k^T W_k]^-1 (Eq. 11)
        post_cov_inv = self.Z_x_inv + info_matrix
        post_cov = torch.linalg.inv(post_cov_inv)
        
        # μ = β * Σ * Σ_k W_k^T y_k (Eq. 12)
        post_mean = self.beta * post_cov @ info_vector
        
        # === Step 3: Compute MARGINAL LIKELIHOOD (Eq. 15) ===
        
        # Reconstruction error: β * Σ_k ||y^(k) - W^(k)μ||²
        recon_error = torch.tensor(0.0, device=self.Z_x.device, dtype=self.Z_x.dtype)
        for k in range(K):
            y_k = y_obs[k]
            W_k = get_W_matrix(
                self.shifts[k:k+1],
                self.rots[k:k+1],
                self.gamma,
                self.grid,
            )
            residual = y_k - W_k @ post_mean  # (M, 1)
            recon_error = recon_error + self.beta * (residual.t() @ residual).squeeze()
        
        # Prior regularization: μ^T Z_x^-1 μ
        prior_term = (post_mean.t() @ self.Z_x_inv @ post_mean).squeeze()
        
        # Log determinant of prior covariance
        sign_Z_x, logdet_Z_x = torch.linalg.slogdet(self.Z_x)
        
        # Log determinant of posterior covariance inverse
        sign_inv, logdet_post_cov_inv = torch.linalg.slogdet(post_cov_inv)
        logdet_post_cov = -logdet_post_cov_inv
        
        # Marginal likelihood (Eq. 15)
        # Note: We return the negative log likelihood for minimization
        neg_log_likelihood = 0.5 * (
            recon_error + 
            prior_term + 
            logdet_Z_x - 
            logdet_post_cov - 
            K * M * torch.log(torch.tensor(self.beta, dtype=self.Z_x.dtype, device=self.Z_x.device))
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