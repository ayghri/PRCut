from pydantic import PositiveInt
from typing import Tuple

from .base import BaseModel


class DatasetConfig(BaseModel):
    dir: str
    name: str
    shape: Tuple[PositiveInt, ...]
    num_classes: PositiveInt
    representation: bool = False


class CIFAR10(DatasetConfig):
    dir: str
    name: str = "cifar10"
    shape: Tuple[PositiveInt, ...] = (3, 32, 32)
    num_classes: PositiveInt = 10


class CIFAR100(DatasetConfig):
    dir: str
    name: str = "cifar100"
    shape: Tuple[PositiveInt, ...] = (3, 32, 32)
    num_classes: PositiveInt = 100


class MNIST(DatasetConfig):
    dir: str
    name: str = "mnist"
    shape: Tuple[PositiveInt, ...] = (28, 28)
    num_classes: PositiveInt = 10


class FashionMNIST(DatasetConfig):
    dir: str
    name: str = "fashion"
    shape: Tuple[PositiveInt, ...] = (28, 28)
    num_classes: PositiveInt = 10
