from typing import Any, Mapping
import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    """
    A wrapper around ResNet models that adds a projection layer as in SimCLR paper.
    """

    model: nn.Module

    def __init__(
        self,
        depth=-1,
        output_dim=512,
        in_channels=3,
        *args,
        **kwargs,
    ) -> None:
        """
        Create ResNet with specified depth and output dimension.

        Parameters:
        depth (int): The depth of ResNet to use
        output_dim (int): The output dimension of the class.
        """
        super().__init__(*args, **kwargs)
        self.depth = depth
        self.output_dim = output_dim
        self.conv = None
        if in_channels != 3:
            self.conv = nn.Conv2d(
                in_channels=in_channels, out_channels=3, kernel_size=1, stride=1
            )
        if depth > 0:
            self.instantiate()

    def instantiate(self):
        assert self.depth in [18, 50]
        if self.depth == 18:
            self.model = models.resnet18(num_classes=self.output_dim)
        else:
            self.model = models.resnet50(pretrained=False, num_classes=self.output_dim)
            # replace the last FC layer
        projection_dim = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(projection_dim, projection_dim), nn.ReLU(), self.model.fc
        )

    def forward(self, inputs):
        if self.conv is not None:
            inputs = self.conv(inputs)
        return self.model(inputs)

    def state_dict(
        self,
        *args,
        destination=None,
        prefix="",
        keep_vars=False,
    ):
        state = super().state_dict(*args, prefix=prefix, keep_vars=keep_vars)
        state["simclr.depth"] = self.depth
        state["simclr.output_dim"] = self.output_dim
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        state_dict = dict(state_dict)
        self.depth = state_dict.pop("simclr.depth")
        self.output_dim = state_dict.pop("simclr.output_dim")
        self.instantiate()
        return super().load_state_dict(state_dict, strict, assign)
