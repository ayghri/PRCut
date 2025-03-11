import torch
from .utils import similarities as sim_utils


class TrueKernel:
    def __init__(self, labels: torch.Tensor) -> None:
        self.labels = labels

    def compute(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        idx_left: None | torch.Tensor = None,
        idx_right: None | torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        y_left = self.labels[idx_left]
        y_right = self.labels[idx_right]
        return torch.eq(y_left.unsqueeze(1), y_right.unsqueeze(0)).to(x_left)


class LocalGaussianKernel(TrueKernel):
    def __init__(self, num_neighbors, factor=1.0):
        self.factor = factor
        self.num_neighbors = num_neighbors

    def compute(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        idx_left: None | torch.Tensor = None,
        idx_right: None | torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return sim_utils.torch_pairwise_similarities(
            x_left, x_right, factor=self.factor
        )


class LocalKnnSimilarity(TrueKernel):
    def __init__(self, num_neighbors, factor=1.0, mode="distance"):
        self.factor = factor
        self.num_neighbors = num_neighbors
        self.mode = mode

    def compute(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        idx_left: None | torch.Tensor = None,
        idx_right: None | torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return sim_utils.torch_knn_similarity(
            x_left=x_left,
            x_right=x_right,
            num_neighbors=self.num_neighbors,
            mode=self.mode,
        )


class EmbeddingSimilarity(TrueKernel):
    def __init__(self, network, num_neighbors, mode="distance", metric="minkowski"):
        self.network = network
        self.num_neighbors = num_neighbors
        self.metric = metric
        self.mode = mode

    def compute(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        idx_left: None | torch.Tensor = None,
        idx_right: None | torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        with torch.no_grad():
            x_left = self.network(x_left)
            x_right = self.network(x_right)
        return sim_utils.torch_knn_similarity(
            x_left=x_left,
            x_right=x_right,
            num_neighbors=self.num_neighbors,
            mode=self.mode,
            metric=self.metric,
        )
