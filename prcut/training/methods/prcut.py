from typing import Dict
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from prcut.losses.prcut import PRCutGradLoss
from prcut.losses.prcut import PRCutBatchLoss
from prcut.configs.trainer import PRCutConfig
from prcut.kernels.types import TrueKernel
from prcut.utils import functional

from .base import BaseTrainer


class PRCutTrainer(BaseTrainer):
    loss_name = "prcut"

    def __init__(
        self,
        config: PRCutConfig,
        network: nn.Module,
        kernel: TrueKernel,
    ):
        super().__init__(config)
        self.config = config
        self.network = network
        self.kernel = kernel
        assert config.batch_size % 2 == 0

        self.optimizer = optim.adam.Adam(
            self.network.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler.lr_decay,
            patience=self.config.scheduler.patience,
        )

        self.loss = PRCutGradLoss()
        self.criterion = PRCutBatchLoss(
            num_clusters=config.num_clusters,
            gamma=config.gamma,
        )

        self.sanity_check()

    def encode(self, x, **kwargs):
        return self.network(x)

    def compute_batch_loss(self, x, x_indices) -> Dict[str, torch.Tensor]:

        W = self.kernel.compute(
            x_left=x[: self.config.batch_size // 2],
            x_right=x[self.config.batch_size // 2 :],
            idx_left=x_indices[: self.config.batch_size // 2],
            idx_right=x_indices[self.config.batch_size // 2 :],
        )
        P = self.encode(x)
        self.criterion.update_cluster_p(P)
        P_l = P[: self.config.batch_size // 2]
        P_r = P[self.config.batch_size // 2 :]

        objective = self.criterion(W, P_l, P_r)
        loss = self.loss(
            W, P_l, P_r, self.criterion.clusters_p, self.config.batch_size
        ) / W.sum() - self.config.entropy_weight * functional.entropy(P.mean(0))

        return {self.loss_name: loss, "objective": objective().item()}
