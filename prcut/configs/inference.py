from typing import List, Tuple
from pydantic import PositiveInt

from .base import BaseModel


class ControllerConfig(BaseModel):
    evaluator: str = "kmeans"
    num_classes: int = 10
    num_clusters: int = 10
    num_neighbors: int = 5
    device_name: str = "cuda"
    validation_metrics: List[str] = ["accuracy", "nmi"]
