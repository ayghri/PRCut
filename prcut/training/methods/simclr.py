import torch
from torch import nn

from prcut.configs.trainer import SimCLRConfig
from prcut.losses.contrastive import SimCLRLoss
from prcut import constructors

from prcut.training.methods.base import BaseTrainer
from prcut.training.augmentations import Transform


class SimCLRTrainer(BaseTrainer):

    loss_name = "cross_entropy"

    def __init__(
        self,
        config: SimCLRConfig,
        encoder: nn.Module,
        augmenter: Transform,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.encoder = encoder
        self.networks["encoder"] = encoder
        self.augmenter = augmenter
        self.device = torch.device(config.device_name)
        self.optimizer = constructors.get_optimizer(
            config.optimizer,
            self.encoder.parameters(),
        )
        self.scheduler = constructors.get_scheduler(config.scheduler, self.optimizer)
        self.criterion = SimCLRLoss(config.temperature)

    def encode(self, x, **kwargs):
        return self.encoder(x)

    def compute_batch_loss(self, x, **kwargs):
        # shape B,D
        z_1 = self.encode(self.augmenter(x))
        z_2 = self.encode(self.augmenter(x))
        return {self.loss_name: self.criterion(z_1, z_2)}
