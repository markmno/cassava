from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import torch
import torchvision.transforms.functional as F


class Transform(ABC):
    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method called to apply selected augmentation
        Args:
            x (torch.Tensor): Input Image

        Returns:
            torch.Tensor: Augmented Image
        """

class Composer:
    def __init__(self, transforms_list: Optional[List[Transform]] = None) -> None:
        """
        Iterator Containing all selected augmentations. 

        Args:
            transforms_list (Optional[List[Transform]], optional): _description_. Defaults to None.
        """
        self.transforms_list = transforms_list if transforms_list is not None else []
        self.transforms_list.insert(0, NoTransform())

    def __iter__(self):
        for transform in self.transforms_list:
            yield transform

    def extend(self, transform_list: List[Transform])->None:
        r"""Extend list of :class:`Transforms` with provided list of augmentations

        Args:
            transform_list (List[Transform]): list of additional augmentations
        """
        self.transforms_list.extend(transform_list)


class NoTransform(Transform):
    """Empty transofrmation

    Args:
        Transform (_type_): _description_
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HFlip(Transform):
    """
    Horizontal Flip
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return F.hflip(x)


class VFlip(Transform):
    """
    Vertical Flip
    """
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return F.vflip(x)


class Rotate(Transform):
    def __init__(self, angle: Literal[90, 180, 270]) -> None:
        """
        Rotating Image

        Args:
            angle (Literal[90, 180, 270]): Angle on which image will be rotated
        """
        super().__init__()
        self.angle = angle

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return F.rotate(x, self.angle)
