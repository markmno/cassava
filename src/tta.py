from abc import ABC, abstractmethod
from typing import List, Literal, Optional
import torch
import torchvision.transforms.functional as F

class Transform(ABC):
    @abstractmethod
    def apply( self, x: torch.Tensor )->torch.Tensor:
        pass
    
class Merger:
    def __init__(self) -> None:
        pass
    
    def __enter__(self):
        pass
    
    def __exit__(self):
        pass
        
class Composer:
    def __init__(self, transforms_list:Optional[List[Transform]]=None) -> None:
        self.transforms_list = transforms_list if transforms_list is not None else []
        self.transforms_list.insert(0, NoTransform())
    
    def __iter__(self):
        for transform in self.transforms_list:
            yield transform

    def extend(self, transform_list:List[Transform]):
        self.transforms_list.extend(transform_list)
    
class NoTransform(Transform):
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
class HFlip(Transform):
    def apply(self,x:torch.Tensor)->torch.Tensor:
        return F.hflip(x)
        
class VFlip(Transform):
    def apply(self, x:torch.Tensor)->torch.Tensor:
        return F.vflip(x)

class Rotate(Transform):
    def __init__(self, angle:Literal[90, 180, 270]) -> None:
        super().__init__()
        self.angle = angle
        
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return F.rotate(x, self.angle)
    
    