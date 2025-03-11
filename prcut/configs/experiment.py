from .base import BaseModel
from .dataset import DatasetConfig
from .trainer import TrainerConfig
from .inference import ControllerConfig


class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    trainer: TrainerConfig
    controller: ControllerConfig
