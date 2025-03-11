from torch.optim import lr_scheduler
from torch.optim.optimizer import ParamsT, Optimizer
from prcut.configs.trainer import OptimizerConfig, SchedulerConfig
from prcut.configs.inference import ControllerConfig
from prcut.configs.dataset import DatasetConfig

from torch import optim
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop
from torch.optim.sgd import SGD
from warnings import warn

STR_OPTIM = ["adam", "rmsprop", "sgd"]
STR_SCHED = ["plateau", "cosine"]


def get_optimizer(config: OptimizerConfig, parameters: ParamsT) -> Optimizer:
    name = config.name
    if name not in STR_OPTIM:
        warn(f"Unknown optimizer '{name}', defaulting to Adam optimizer")
        warn(f"Available optimizers: {STR_OPTIM}")

    if name == "rmsprop":
        return RMSprop(
            parameters,
            lr=config.lr,
            alpha=config.alpha,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )

    if name == "sgd":
        return SGD(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            dampening=config.dampening,
        )
    return Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)


def get_scheduler(config: SchedulerConfig, optimizer) -> lr_scheduler.LRScheduler:
    name = config.name
    if name not in STR_SCHED:
        warn(f"Unknown optimizer '{name}', defaulting to constant scheduler")
        warn(f"Available schedulers: {STR_SCHED}")
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.lr_decay,
            patience=config.patience,
        )
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.t_max,
            eta_min=config.eta_min,  # TODO: check meaning
            last_epoch=config.last_epoch,
        )
    return lr_scheduler.LRScheduler(optimizer)


def get_embedding_evaluator(config: ControllerConfig):
    if config.evaluator == "spectral":
        from sklearn.cluster import SpectralClustering

        return SpectralClustering(
            n_clusters=config.num_clusters, n_neighbors=config.num_neighbors
        )

    from sklearn.cluster import KMeans

    return KMeans(n_clusters=config.num_clusters)


def get_dataset(config: DatasetConfig, train=True):
    name = config.name
    assert name in ["mnist", "fashion", "cifar10", "cifar100"]
    from prcut.data.data_utils import get_mnist, get_cifar10, get_fashionmnist
    from prcut.data.data_utils import get_cifar100

    if name == "mnist":
        return get_mnist(data_dir=config.dir, train=train)
    if name == "cifar10":
        print("Getting CIFAR10 dataset")
        return get_cifar10(data_dir=config.dir, train=train)
    if name == "cifar100":
        print("Getting CIFAR100 dataset")
        return get_cifar100(data_dir=config.dir, train=train)
    print("Getting fashion-mnist dataset")
    return get_fashionmnist(data_dir=config.dir, train=train)
