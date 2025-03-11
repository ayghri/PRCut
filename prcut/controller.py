import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Union
import torch
from .configs.inference import ControllerConfig
from .metrics import evaluate_clustering
from . import constructors


class ControllerBase(ABC):
    def __init__(self, model: torch.nn.Module, config: ControllerConfig):
        self.model = model
        self.config = config
        self.device = torch.device(self.config.device_name)

    @abstractmethod
    def compute_outputs(
        self, data_loader
    ) -> Union[Tuple[Any, Any], Tuple[Any, Any, Any]]: ...

    @abstractmethod
    def validate(self, data_loader) -> Dict: ...


class EmbeddingController(ControllerBase):

    def compute_outputs(self, data_loader, return_features=False):
        features = []
        embeddings = []
        labels = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device)
                if return_features:
                    features.append(x.cpu().numpy())
                embeddings.append(self.model(x).cpu().numpy())
                labels.append(y.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        if return_features:
            features = np.concatenate(features, axis=0)
            return embeddings, labels, features

        return embeddings, labels

    def compute_assignment(self, data_loader):
        embeddings, labels = self.compute_outputs(data_loader)[:2]
        pred_labels = constructors.get_embedding_evaluator(self.config).fit_predict(
            embeddings
        )
        return pred_labels, labels

    def validate(self, data_loader, prefix=""):
        pred_labels, labels = self.compute_assignment(data_loader)
        prefix = "" if len(prefix) == 0 else f"{prefix}_"
        return {
            f"{prefix}{k}": v
            for k, v in evaluate_clustering(
                true_labels=labels,
                cluster_assignments=pred_labels,
                num_classes=self.config.num_classes,
                num_clusters=self.config.num_clusters,
            ).items()
        }


class ClusterController(EmbeddingController):

    def compute_assignment(self, data_loader):
        embeddings, labels = self.compute_outputs(data_loader)[:2]
        pred_labels = embeddings.argmax(1)
        return pred_labels, labels
