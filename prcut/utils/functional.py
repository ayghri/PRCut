from typing import Optional, Tuple, Union
import torch
import numpy as np
from scipy.special import roots_legendre


def entropy(p: torch.Tensor) -> torch.Tensor:
    return torch.special.entr(p)


def sample_excluded(n: int, k: int, a: int, b: int) -> torch.Tensor:
    """
    Generate a subset of size k from the range [0, n-2] excluding the values a and b.

    Parameters:
    n (int): The upper limit of the range.
    k (int): The size of the subset to generate.
    a (int): The first value to exclude.
    b (int): The second value to exclude.

    Returns:
    torch.Tensor: A subset of size k excluding the values a and b.
    """
    if a > b:
        return sample_excluded(n, k, b, a)
    elif a == b:
        raise ValueError(f"a and b has to be different, got {a, b}")
    else:
        subset = np.random.choice(n - 2, size=k, replace=False)
        subset = subset + (subset >= a)
        subset = subset + (subset >= b)
        return torch.tensor(subset).long()


def sample_integral_subset(
    batch_size: int,
    indices_left: torch.Tensor,
    indices_right: torch.Tensor,
    subset_size: int,
) -> torch.Tensor:
    """
    Compute batches to use for integrals based on the given pairs of left and right indices.

    Parameters:
    - batch_size: int, usually B=2*A
    - left_indices: np.ndarray, array of left indices (A,)
    - right_indices: np.ndarray, array of right indices (A,)
    - subset_size: int, size of the subset, subset_size <= B-2

    Returns:
    - indices: torch.Tensor, tensor containing the computed batches , (A,A,s)
    """
    assert indices_left.shape[0] == indices_right.shape[0]
    assert batch_size >= indices_left.shape[0]
    assert batch_size >= subset_size + 2
    indices = torch.zeros(
        (indices_left.shape[0], indices_right.shape[0], subset_size), dtype=torch.long
    )
    for i in range(indices.shape[0]):
        for j in range(indices.shape[0]):
            indices[i, j] = sample_excluded(
                batch_size,
                subset_size,
                int(indices_left[i].item()),
                int(indices_right[j].item()),
            )
    return indices


def sum_excluding_self(t_array: torch.Tensor):
    to_keep_l = torch.logical_not(
        torch.eye(t_array.shape[0]).to(t_array.device, dtype=torch.bool)
    )
    indices = torch.where(to_keep_l)[1].view(t_array.shape[0], -1)
    return t_array[indices].sum(1)


def legendre_quadrature(a, b, degree) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the roots and the weights for Legendre quadrature on interval [a,b]
    Args:
        a: integral left limit
        b: integral right limit
        degree: polynomial degree
    Returns: tuple (roots, weights)
    """
    assert a < b
    roots, weights = roots_legendre(degree)
    weights = weights * (b - a) / 2
    roots = (b - a) / 2 * roots + (a + b) / 2
    return torch.tensor(roots).float(), torch.tensor(weights).float()


def batch_quadrature(integral_batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses Legendre to integrate a polynomial of degree batch_size on [0,1]
    Args:
        batch_size: also the number of probabilities used in the product
    Returns:
        (roots, weights)
    """
    return legendre_quadrature(0, 1, integral_batch_size // 2 + 1)


def estimate_quadrature(
    p: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    integral_subset: Union[torch.Tensor, None] = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
        Compute the integral of a function using quadrature.
    Args:
        p: Tensor (B, k)
        indices_left: Tensor (a,) containing indices i of left probabilities
        indices_right: Tensor (a,) containing indices k of right probabilities
        subset_size: int, s
        roots: Tensor (T,)
        weights: Tensor(T,)
    Returns: Tensor (B, k) of elements I(i,k) estimated using s elements
    """
    # (a, s)
    # if integral_subset is None:
    # integral_subset = sample_integral_subset(
    # p.shape[0], indices_left, indices_right, subset_size
    # )
    # (a, a, s, k)
    if integral_subset is None:
        # (a,k, 1)
        p_integral = p.unsqueeze(-1)
        # (a,k,T)
        integral_estimate = torch.log(1 - p_integral * roots.view(1, 1, -1))
        # (k,T)
        log_integrals = integral_estimate.sum(0) + torch.log(weights.view(1, -1))
        # (1, 1, K,T)
        log_integrals = log_integrals.unsqueeze(0).unsqueeze(0)
    else:
        # (a,a,s,k)
        p_integral = p[integral_subset]
        # (a,a,s,k,T)
        integral_estimate = torch.log(
            1 - p_integral.unsqueeze(-1) * roots.view(1, 1, 1, 1, -1)
        )
        # (a,a,k,T)
        log_integrals = integral_estimate.sum(2) + torch.log(weights.view(1, 1, 1, -1))
    # (a,a,k)
    return torch.exp(log_integrals).sum(-1) ** gamma


def compute_prcut_loss(
    p: torch.Tensor,
    indices_left: torch.Tensor,
    indices_right: torch.Tensor,
    similarities: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    integral_subset: Union[torch.Tensor, None] = None,
    gamma: float = 1.0,
) -> torch.Tensor:

    p_l = p[indices_left].unsqueeze(1)
    # (1,A,k)
    p_r = p[indices_right].unsqueeze(0)
    # (A,k) : w_ik * (p_ij + p_kj -2p_ij p_kj)
    # (A,A,k)
    multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r)
    # multipliers = similarities.unsqueeze(-1) * (p_l - p_r)**2

    # (A,A, K)
    right_quantity = estimate_quadrature(
        p=p,
        roots=roots,
        weights=weights,
        integral_subset=integral_subset,
        gamma=gamma,
    )
    # print("compute_prcut", multipliers.shape, right_quantity.shape)
    return (multipliers * right_quantity).sum()


# def compute_masked_prcut_loss(
#     p_l: torch.Tensor,
#     p_r: torch.Tensor,
#     similarities: torch.Tensor,
#     gamma: float = 1.0,
#     reduce: str = "sum",
# ) -> torch.Tensor:
#
#     p_l = p.unsqueeze(1)
#     # (1,A,k)
#     p_r = p.unsqueeze(0)
#     # (A,k) : w_ik * (p_ij + p_kj -2p_ij p_kj)
#     # (A,A,k)
#     if reduce == "sum":
#         multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r).sum(
#             [0, 1]
#         )
#     else:
#         multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r).mean(
#             [0, 1]
#         )
#         # right_quantity = integral_quadrature(probs=p_quad, gamma=gamma)
#
#     return (multipliers * right_quantity).sum()
#


def compute_simple_prcut_loss(
    p: torch.Tensor,
    similarities: torch.Tensor,
    p_quad: torch.Tensor,
    gamma: float = 1.0,
    reduce: str = "sum",
) -> torch.Tensor:

    p_l = p.unsqueeze(1)
    # (1,A,k)
    p_r = p.unsqueeze(0)
    # (A,k) : w_ik * (p_ij + p_kj -2p_ij p_kj)
    # (A,A,k)
    if reduce == "sum":
        multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r).sum(
            [0, 1]
        )
    else:
        multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r).mean(
            [0, 1]
        )
    right_quantity = integral_quadrature(probs=p_quad, gamma=gamma)

    return (multipliers * right_quantity).sum()


def integral_quadrature(probs: torch.Tensor, gamma: float = 1.0):
    b_size = probs.shape[0]
    roots, weights = batch_quadrature(b_size)
    # shape (T,)
    roots = roots.to(probs)
    # shape (T,)
    weights = weights.to(probs)
    # shape (B,K, 1)
    p_integral = probs.unsqueeze(-1)
    # shape (B,K,T)
    integral_estimate = torch.log(1 - p_integral * roots.view(1, 1, -1))
    # (K,T)
    log_integrals = integral_estimate.sum(0) + torch.log(weights.view(1, -1))
    # (K, T)
    log_integrals = log_integrals
    # (K,)
    return torch.exp(log_integrals).sum(-1) ** gamma


def simple_quadrature_estimation(
    p: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    # (A,K,1)
    p_integral = p.unsqueeze(-1)
    # (A,K,T)
    integral_estimate = torch.log(1 - p_integral * roots.view(1, 1, -1))
    # (K,T)
    log_integrals = integral_estimate.sum(0) + torch.log(weights.view(1, -1))
    # (1, 1, K, T)
    log_integrals = log_integrals.unsqueeze(0).unsqueeze(0)
    # (1,1,K)
    return torch.exp(log_integrals).sum(-1) ** gamma


def pairwise_quadrature(p_l: torch.Tensor, p_r: torch.Tensor, gamma: float = 1.0):
    bl_size = p_l.shape[0]
    br_size = p_r.shape[0]
    # assert bl_size == br_size, "both probs should have same size"
    roots, weights = batch_quadrature(bl_size + br_size - 2)
    # shape (1, 1, T)
    roots = roots.to(p_l).view(1, 1, -1)
    # shape (1, T)
    weights = weights.to(p_l).view(1, 1, 1, -1)
    # shape (Bl,K,T)
    log_p_l_1 = sum_excluding_self(torch.log(1 - p_l.unsqueeze(-1) * roots))
    # shape (Br,K,T)
    log_p_r_1 = sum_excluding_self(torch.log(1 - p_r.unsqueeze(-1) * roots))
    # (K,T)
    # sum_estimate = sum_excluding_self(log_p_l_1) + sum_excluding_self(log_p_r_1)
    # (Bl, Br, K, T)
    log_p_pairwise = log_p_l_1.unsqueeze(1) + log_p_r_1.unsqueeze(0)
    return torch.exp(log_p_pairwise).mul(weights).sum(-1) ** gamma


def graph_integral_quadrature(probs: torch.Tensor, edge_index, gamma: float = 1.0):
    b_size = probs.shape[0]
    roots, weights = batch_quadrature(b_size)
    # shape (T,)
    roots = roots.to(probs)
    # shape (T,)
    weights = weights.to(probs)
    # shape (B,K, 1)
    return graph_quadratue(probs, edge_index, roots, weights, gamma)


def graph_quadratue(
    p,
    edge_index,  # shape (2,E)
    roots,
    weights,
    gamma,
):
    # B,K, T
    log_1_pt = torch.log(1 - p.unsqueeze(-1) * roots.view(1, 1, -1))
    # E, K, T
    batch_quad = log_1_pt.sum(0, keepdim=True) - log_1_pt[edge_index].sum(dim=0)
    # E, K
    return (torch.exp(batch_quad) * weights.view(1, 1, -1)).sum(-1) ** gamma


def graph_prcut(probs, edge_index, gamma=1.0):

    integral = graph_integral_quadrature(probs, edge_index, gamma)
    p = probs[edge_index]
    multipliers = (p[0] + p[1] - 2 * p[0] * p[1]) * integral
    return multipliers.mean()


def pairwise_prcut(p_l, p_r, similarities, gamma=1.0):
    # (Bl,Br, K)
    quadrature = pairwise_quadrature(p_l, p_r, gamma=gamma)
    p_l = p_l.unsqueeze(1)
    p_r = p_r.unsqueeze(0)
    # (Bl, Br, K)
    multipliers = p_l + p_r - 2 * p_l * p_r
    return (similarities * (multipliers * quadrature).sum(-1)).mean()


def coupled_quadrature_estimation(
    p_left: torch.Tensor,
    p_right: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    # (A,1,K)
    p_l = p_left.unsqueeze(1)
    # (1,B,K)
    p_r = p_right.unsqueeze(0)
    # (A,1,K,T)
    log_p1_l = torch.log(1 - p_l.unsqueeze(-1) * roots.view(1, 1, -1))
    # (1,B,K,T)
    log_p1_r = torch.log(1 - p_r.unsqueeze(-1) * roots.view(1, 1, -1))
    # (A,B,K,T)
    integral_estimate = (
        log_p1_l.sum(0, keepdim=True)
        + log_p1_r.sum(1, keepdim=True)
        - log_p1_l
        - log_p1_r
    )
    # (A,B,K,T)
    log_integrals = integral_estimate + torch.log(weights.view(1, 1, 1, -1))
    # (1,1,K)
    return torch.exp(log_integrals).sum(-1) ** gamma


def compute_decoupled_prcut_loss(
    p_left: torch.Tensor,
    p_right: torch.Tensor,
    similarities: torch.Tensor,
    p_integral: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:

    # (B1,1,K)
    p_l = p_left.unsqueeze(1)
    # (1,B2,K)
    p_r = p_right.unsqueeze(0)
    # (B1, B2, K)
    multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r)
    # (A,A, K)
    right_quantity = simple_quadrature_estimation(
        p=p_integral,
        roots=roots,
        weights=weights,
        gamma=gamma,
    )
    # print("compute_prcut", multipliers.shape, right_quantity.shape)
    return (multipliers * right_quantity).sum()


def compute_coupled_prcut_loss(
    p_left: torch.Tensor,
    p_right: torch.Tensor,
    similarities: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:

    # (B1,1,K)
    p_l = p_left.unsqueeze(1)
    # (1,B2,K)
    p_r = p_right.unsqueeze(0)
    # (B1, B2, K)
    multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r)
    # (A,A, K)
    right_quantity = coupled_quadrature_estimation(
        p_left=p_left,
        p_right=p_right,
        roots=roots,
        weights=weights,
        gamma=gamma,
    )
    # print("compute_prcut", multipliers.shape, right_quantity.shape)
    return (multipliers * right_quantity).sum()


def masked_softmax(logits, mask, dim=-1) -> torch.Tensor:
    logits = logits.masked_fill(mask, float("-inf"))
    return torch.nn.functional.softmax(logits, dim=dim)


def noisy_softmax(logits, noise_scale=1.0, tau=1.0, dim=-1) -> torch.Tensor:
    return torch.softmax(
        (logits + noise_scale * torch.randn_like(logits)) / tau, dim=dim
    )


def topk_softmax(
    logits: torch.Tensor,
    k: int = 2,
    dim: int = -1,
) -> torch.Tensor:
    _, indices = torch.topk(logits, k, dim=dim)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(index=indices, value=False, dim=-1)
    return masked_softmax(logits, mask, dim=dim)
