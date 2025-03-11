from pydantic import PositiveInt
from .base import BaseModel
from typing import Tuple, Literal


class OptimizerConfig(BaseModel):
    name: str = "adam"
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_decay: float = 0.99
    weight_decay: float = 0.0
    min_lr: float = 1e-6
    alpha: float = 0.99  # rmsprop
    momentum: float = 0.0  # rmsprop, sgd
    dampening: float = 0.0  # sgd


class SchedulerConfig(BaseModel):
    name: str = "none"
    mode: Literal["min", "max"] = "min"
    lr_decay: float = 0.95
    lr_min: float = 0.0
    eta_min: int = 50
    patience: int = 5
    t_max: int = 100
    last_epoch: int = -1


class TrainerConfig(BaseModel):
    log_dir: str
    device_name: str = "cpu"
    n_epochs: PositiveInt = 2
    start_epoch: int = 0
    batch_size: PositiveInt = 64
    num_clusters: int = 10
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    autocast: bool = False
    gradient_scaling: bool = False


class AEConfig(TrainerConfig): ...


# self.device = device
# self.ae_config = config
# self.lr = self.ae_config["lr"]
# self.epochs = self.ae_config["epochs"]
# self.min_lr = self.ae_config["min_lr"]
# self.lr_decay = self.ae_config["lr_decay"]
# self.patience = self.ae_config["patience"]
# self.architecture = self.ae_config["hiddens"]
# self.batch_size = self.ae_config["batch_size"]
# self.weights_path = "spectralnet/_trainers/weights/ae_weights.pth"


class VAEConfig(TrainerConfig): ...


class VaDEConfig(TrainerConfig):
    pre_n_epochs: int = 10
    num_recon_samples: int = 1
    eps: float = 1e-8
    latent_dim: int
    output_dim: int


class ClusterGANConfig(TrainerConfig):
    metric_type: str = "wass"  # or "van"
    num_workers: int = 4

    test_batch_size: int = 5000
    lr: float = 1e-4
    b1: float = 0.5
    b2: float = 0.9  # 99
    decay: float = 2.5 * 1e-5
    n_skip_iter: int = 1  # 5

    img_size: int = 28
    channels: int = 1

    # Latent space info
    latent_dim: int = 30
    n_c: int = 10
    betan: int = 10
    betac: int = 10


class SiameseConfig(TrainerConfig):
    num_neighbors: int = 10
    use_approximation: bool = False


class SpectralNetConfig(TrainerConfig):
    is_sparse: bool = False
    lr: float = 0.001
    n_nbg: int = 5
    min_lr: float = 0.0001
    scale_k: int = 1
    lr_decay: float = 0.1
    patience: int = 10
    batch_size: int = 32
    is_local_scale: bool = True
    similarity: str = "local"


class PRCutConfig(TrainerConfig):
    subset_size: int = 32
    gamma: float = 2.0
    noise: float = 0.5
    integral_size: int = 512
    num_neighbors: int = 2


class SimCLRConfig(TrainerConfig):
    n_views: int = 2
    temperature: float = 1.0
    resnet_depth: int = 18
    embedding_dim: int = 512


class TurtleConfig(TrainerConfig):
    gamma: float = 10.0  # entropy regularization weight
    num_iter: int = 6000
    num_inner_iter: int = 60
    inner_optim: OptimizerConfig
    outer_optim: OptimizerConfig
    batch_size: int = 10000
    warm_start: bool = False
