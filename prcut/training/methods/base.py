from typing import Dict, Any, Callable, Union, Any, Generic, TypeVar
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim.optimizer import Optimizer
from prcut.configs.trainer import TrainerConfig


SomeConfig = TypeVar("SomeConfig", bound="TrainerConfig")


class BaseTrainer(ABC, Generic[SomeConfig]):
    optimizer: Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_name: str
    monitor_metric: str
    config: SomeConfig

    def __init__(
        self,
        config: SomeConfig,
        batch_callback: Union[Callable[[Any, torch.Tensor], None], None] = None,
    ):
        self.device = torch.device(config.device_name)
        self.networks: Dict[str, torch.nn.Module] = {}
        self.config = config

        self.batch_callback = batch_callback
        self.ckpt_path = Path(config.log_dir).joinpath("checkpoints")

        self.ckpt_path.mkdir(parents=True, exist_ok=True)

    def sanity_check(self):
        assert self.loss_name is not None
        assert self.optimizer is not None

    def checkpoint(self, epoch):
        for name, net in self.networks.items():
            ckpt_name = f"{name}_{epoch}.ckpt"
            print("Saving", name, "to", ckpt_name)
            torch.save(net.state_dict(), self.ckpt_path.joinpath(ckpt_name))

    # @abstractmethod
    def compute_batch_loss(self, x, **kwargs) -> Dict[str, torch.Tensor]:
        return {}

    # @abstractmethod
    def encode(self, x, **kwargs):
        pass

    def compute_outputs(self, x, **kwargs):
        return self.encode(x, **kwargs)

    def optimize_loss(self, loss, retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

    def train_step(self, x, retain_graph=False) -> Dict[str, torch.Tensor]:
        loss = self.compute_batch_loss(x)
        self.optimize_loss(loss[self.loss_name], retain_graph=retain_graph)
        if self.batch_callback is not None:
            self.batch_callback(self, loss[self.loss_name])
        return loss

    def train_epoch(self, dataloader, start_step=0):
        total_loss: Dict[str, torch.Tensor] = defaultdict(lambda: torch.tensor(0.0))
        num_batch = start_step
        for (x,) in tqdm(dataloader):
            loss = self.train_step(x.to(self.device))
            for name, value in loss.items():
                total_loss[name] += value.detach().cpu()
            # total_loss += loss[self.loss_name].detach().cpu().numpy().item()
            num_batch = num_batch + 1
        for name, value in total_loss.items():
            total_loss[name] = value / (num_batch - start_step)
        return total_loss

    def update_scheduler(self, monitored_value):
        self.scheduler.step(monitored_value)
        return self.get_lr()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]
