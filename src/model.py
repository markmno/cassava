from typing import Tuple

import timm
import torch
import torch.nn as nn

model_list: Tuple[str] = tuple(timm.list_models())

class TimmModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True)->None:
        """
        Wrapper around timm lirary to create models.

        Args:
            model_name (str): Name of model architechture
            pretrained (bool, optional): Pretrained state. Defaults to True.
        """
        super().__init__()
        if model_name not in model_list:
            raise KeyError
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=5
        )

    def forward(self, x)->torch.Tensor:
        return self.model(x)
