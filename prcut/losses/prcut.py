from typing import Tuple
import torch
from torch import nn


@torch.no_grad()
def offline_gradient(W: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    ov_P = P.mean(0)
    left = (W.sum(0).unsqueeze(1) - 2 * torch.mm(W, P.detach())) / ov_P
    right = -(W.mm(P) - W.mm(P) * P).sum(0) / ov_P**2 / P.size(0)
    P_grad = left + right
    return P_grad


@torch.no_grad()
def batch_cluster_prcut_loss(
    W: torch.Tensor,
    P_l: torch.Tensor,
    P_r: torch.Tensor,
    ov_P: torch.Tensor,
) -> torch.Tensor:
    P_l = P_l.unsqueeze(1)
    P_r = P_r.unsqueeze(0)
    return (W.unsqueeze(-1) * (P_l + P_r - 2 * P_l * P_r)).sum(dim=(0, 1)) / ov_P


@torch.no_grad()
def batch_gradient(
    W: torch.Tensor,
    P_l: torch.Tensor,
    P_r: torch.Tensor,
    ov_P: torch.Tensor,
    n: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the PRCut batch gradient for a given weight matrix and probabilities

    Parameters:
    W (torch.Tensor): Weight matrix of shape (a, b).
    P_l (torch.Tensor): Left probability distribution of shape (a, k).
    P_r (torch.Tensor): Right probability distribution of shape (b, k).
    ov_P (float): Clusters likelihood, shape (k,)
    n (int): Number of samples.

    Returns:
    tuple: A tuple containing two torch.Tensors:
        - Gradient with respect to the left probabilities (shape: (a, k)).
        - Gradient with respect to the right probabilities (shape: (b, k)).
    """
    left_l = W.mm(1 - 2 * P_r) / ov_P
    left_r = W.t().mm(1 - 2 * P_l) / ov_P
    right = -batch_cluster_prcut_loss(W, P_l, P_r, ov_P) / ov_P / n
    return left_l + right, left_r + right


class PRCutGradLoss(nn.Module):
    """
    Computes the PRCut Loss that can be backpropagated.
    This is different from the PRCut batch estimator: See PRCutBathLoss

    Parameters:
        W (Tensor): Weight tensor, shape (a,b)
        P_l (Tensor): Left probs tensor, shape (a,k)
        P_r (Tensor): Right probs tensor, shape (b,k)
        ov_P (Tensor): Cluster likelihood estimator (k,)
        n (int): Number of samples.

    Returns:
        Tensor: The computed PRCut Loss for backpropagation
    """

    def forward(self, W, P_l, P_r, ov_P, n) -> torch.Tensor:
        P_l_grad, P_r_grad = batch_gradient(W, P_l, P_r, ov_P, n)
        return (P_l_grad * P_l).sum() + (P_r_grad * P_r).sum()


class PRCutBatchLoss(nn.Module):
    """
    Computes the PRCut batch estimate given the estimator clusters likelihood
    This is used to get an unbiased estimate of the PRCut and monitor it during training

    Parameters:
        W (Tensor): Weight tensor, shape (a,b)
        P_l (Tensor): Left probs tensor, shape (a,k)
        P_r (Tensor): Right probs tensor, shape (b,k)
        ov_P (Tensor): Cluster likelihood estimator (k,)

    Returns:
        Tensor: unbiased batch estimate of PRCut
    """

    def __init__(self, num_clusters, gamma, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_clusters = num_clusters
        self.gamma = gamma
        self.clusters_p = nn.Parameter(torch.ones(num_clusters) / num_clusters)
        self.register_parameter(name="cluster_p", param=self.clusters_p)

    def update_cluster_p(self, P):
        self.clusters_p.data.mul_(1 - self.gamma)
        self.clusters_p.data.add_(P.mean(0).detach() * self.gamma)

    @torch.no_grad()
    def forward(self, W, P_l, P_r) -> torch.Tensor:
        P_i = P_l.unsqueeze(1)
        P_j = P_r.unsqueeze(0)
        return (
            (W.unsqueeze(-1) * (P_i + P_j - 2 * P_i * P_j)).sum(dim=(0, 1))
            / self.clusters_p
        ).sum()


class SimplexL2Loss(nn.Module):
    """
    Returns a Simplex L2 loss that yields an unbiased gradient
    The original loss is sum_l (p_l-1/k)^2, where p_l = (sum_i^n p_(i,l))/n

    The gradient is 2(p_l-1/k)/n, so we need an estimator of p_l, which is ov_P
    ov_P is the cluster likelihood

    """

    def forward(self, P, ov_P) -> torch.Tensor:
        return ((ov_P - 1 / P.size(1)) * P.mean(0)).sum()
