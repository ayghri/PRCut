import torch
from torch import nn


class VAEBCELoss(nn.Module):
    def __init__(self, decoder, num_samples=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bce = nn.BCELoss()
        self.decoder = decoder
        self.num_samples = num_samples

    def forward(self, real_x, z_mu, z_log_sigma2):
        loss = 0.0
        for _ in range(self.num_samples):
            z = z_mu + torch.randn_like(z_mu) * torch.exp(z_log_sigma2 / 2)
            x_recon = self.decoder(z)  # has to return values in [0.1]
            loss += self.bce(x_recon, real_x)
        return loss / self.num_samples


class VaDeELBOLoss(nn.Module):
    def __init__(self, decoder, num_samples=1, det=1e-10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.det = det
        self.vae_bce = VAEBCELoss(decoder, num_samples=num_samples)

    def forward(self, z_mu, z_sigma2_log, pi, mu_c, log_sigma2_c, gaussian_pdfs_log):
        """
        Compute the ELBO loss of the Vade model.

        Args:

        z_mu (torch.Tensor)(J,D): The mean of the latent space.
        z_sigma2_log (torch.Tensor) (J,D): The log of the variance of the latent space.
        pi (torch.Tensor)(K): The prior probabilities of the clusters.
        mu_c (torch.Tensor)(K,D): The means of the clusters.
        log_sigma2_c (torch.Tensor)(K,D): The log of the variances of the clusters.

        """
        loss_total = 0.0

        z = z_mu + torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2)
        cluster_log_probs = torch.log(pi.unsqueeze(0)) + gaussian_pdfs_log
        # (J, K)
        cluster_probs = torch.exp(cluster_log_probs) + self.det

        # (J, K)
        cluster_probs = cluster_probs / cluster_probs.sum(1, keepdim=True)

        # (sigma ratio) (J, K, D)
        sigma_ratio = torch.exp(z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(0))
        # mu|j - mu_c|j / sigma_c|j, (J, K, D)
        mu_delta = (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) / torch.exp(
            log_sigma2_c.unsqueeze(0)
        )
        # (J, K)
        sum_terms = (log_sigma2_c.unsqueeze(0) + sigma_ratio + mu_delta).sum(2)
        loss_total = 0.5 * torch.sum(cluster_probs * sum_terms, 1).sum()

        loss_total -= torch.mean(
            torch.sum(cluster_probs * torch.log(pi.unsqueeze(0) / (cluster_probs)), 1)
        ) + 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))

        return loss_total + self.vae_bce(z, z_mu, z_sigma2_log)


class MutualInfoLoss(nn.Module):
    def __init__(self, eps=1e-8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        p_hat = p.mean(0)
        p_hat = (p_hat + self.eps) / (1 + p_hat.size(0) * self.eps)
        p_1 = p_hat.unsqueeze(1)
        p_2 = p_hat.unsqueeze(0)
        mi = -entropy(1 - p_1 - p_2) + entropy(1 - p_1) + entropy(1 - p_2)
        return mi.sum()


class KLDLoss(nn.Module):
    def __init__(self, mu_ref: float = 0.0, var_ref=0.0):
        super(KLDLoss, self).__init__()
        self.mu_ref = mu_ref
        self.var_ref = var_ref

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor):
        """
        Compute the Kullback-Leibler Divergence loss for the given mean and log variance.

        Args:
        mu (torch.Tensor): Mean of the latent Gaussian [B x D]
        log_var (torch.Tensor): Log variance of the latent Gaussian [B x D]

        Returns:
        torch.Tensor: KLD loss
        """
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        return kld_loss
