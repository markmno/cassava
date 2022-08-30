from typing import Any, List
from .base import IHook

class TTAHook(IHook):
    def __init__(self, augmentations:List[Any]) -> None:
        super().__init__()
        self.augmentations = augmentations
    
    def before_run(self):
        return super().before_run()
    
    def before_eval_step(self):
        for augmentation in self.augmentations:
            pass
        
        
