import torch
from torch import nn


class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(
        self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False
    ) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet model.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W,
        and the orthonormalized output of the network.

        Args:
            W (torch.Tensor):               Affinity matrix
            Y (torch.Tensor):               Output of the network
            is_normalized (bool, optional): Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        m = Y.size(0)
        if is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / D[:, None]

        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, label: torch.Tensor, p: int = 2
    ) -> torch.Tensor:
        """
        Compute the contrastive loss between the two outputs of the siamese network.

        Parameters
        ----------
        x1 : torch.Tensor (B,d)
            The first output of the siamese network.
        x2 : torch.Tensor (B,d)
            The second output of the siamese network.
        label : torch.Tensor (B,)
            label[i] indicates whether x1[i] and x2[i] are similar (1) or not (0).
        p : int, optional
            The norm to use for the distance calculation. Default is 2.

        Returns
        -------
        torch.Tensor
            The computed contrastive loss value.
        """

        # shape (B,)
        euclidean = nn.functional.pairwise_distance(x1, x2, p=p)

        positive_distance = torch.pow(euclidean, p)
        negative_distance = torch.pow(torch.clamp(self.margin - euclidean, min=0.0), p)

        loss = torch.mean(
            (label * positive_distance) + ((1 - label) * negative_distance)
        )
        return loss


class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def forward(self, z_1, z_2):
        assert z_1.size() == z_2.size(), "z_1,z_2 are representations of the same input"
        # print(z_1.shape, z_2.shape)
        B, _ = z_1.shape  # shape = B, K
        sample_index = torch.remainder(torch.arange(B * 2), B)
        # shape (2B, 2B), equal 1 if z[i],z[j] represent the same sample, 0 otherwise
        labels = (sample_index.unsqueeze(0) == sample_index.unsqueeze(1)).float()
        z = torch.cat([z_1, z_2])
        # shape (2B, K)
        z = nn.functional.normalize(z, dim=1)
        # shape (2B, 2B)
        similarity_matrix = torch.matmul(z, z.T)
        # masked used to discard the main diagonal from y and similarity_matrix since
        # self-similarity is always equal = 1
        # shape (2B, 2B)
        to_keep = torch.logical_not(
            torch.eye(labels.shape[0]).to(labels.device, dtype=torch.bool)
        )
        # shape(2B, 2B-1)
        labels = labels[to_keep].view(labels.shape[0], -1)
        # shape(2B, 2B-1)
        similarity_matrix = similarity_matrix[to_keep].view(
            similarity_matrix.shape[0], -1
        )
        # (2B,1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # (2B, 2B-2)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )
        # (2B, 2B-1) with positive logits at [:,0]
        logits = torch.cat([positives, negatives], dim=1)
        # labels 0 for everything
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        # scale by temperature
        logits = logits / self.temperature
        return self.criterion(logits, labels)
