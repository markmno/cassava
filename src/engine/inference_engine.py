from typing import List
import torch
import torch.nn as nn
import numpy as np
from .base import IEngine
from ..tta import Composer, HFlip, NoTransform, Rotate, VFlip
from ..hooks.progressbar_hook import EpochProgress, EvalStepProgress, InferenceFoldProgress, InferenceStepProgress, TrainStepProgress
from ..hooks import InferenceHookList, InferenceProgressBarHook
from ..storage import Storage
from src import storage
from rich.traceback import install

install(show_locals=False)

device = torch.device('cuda')

__all__ = ["InferenceEngine", "EnsembleEngine"]

class InferenceEngine(IEngine):
    """_summary_

    Args:
        IEngine (_type_): _description_
    """
    def __init__(self, model:nn.Module, weights:List[str]|str) -> None:
        """_summary_

        Args:
            model (nn.Module): _description_
            weights (List[str] | str): _description_
        """
        super().__init__()
        model.eval()
        self.model = model.to(device)
        self.weights_list = weights 
        
    def run(self)->np.ndarray:
        self.storage.put('eval_steps', len(self.dataloader))
        self.storage.put('folds', len(self.weights_list))
        self.hooks.before_run()
        fold_preds = []
        for weights in self.weights_list:
            self.hooks.before_epoch()
            self.model.load_state_dict(torch.load(weights))
            fold_preds += [self.inference_step()]
            self.hooks.after_epoch()
        self.hooks.after_run()
        return np.mean(fold_preds, 0)
    
    @torch.no_grad()
    def inference_step(self):
            preds = []
            for image in self.dataloader:
                self.hooks.before_inference_step()
                image = image.to(device)
                aug_images = self.augment(image)
                tta_preds = []
                for aug_image in aug_images:
                    logits = self.model(aug_image)
                    tta_preds += [torch.softmax(logits, 1)] #normalize outputs
                self.hooks.after_inference_step()
                preds += [torch.cat(tta_preds, 0).mean(0, keepdim=True)]
            return torch.cat(preds, dim=0).detach().cpu().numpy()
     
    def augment(self, image:torch.Tensor):
            img_list = []
            for transforms in self.transforms:
                img_list.append(transforms.apply(image))
            return img_list 
        
    def build(self, dataloader):
            self.dataloader = dataloader
            self.storage = Storage()
            self.transforms = self.build_transforms()
            self.hooks = self.build_hooks()
            self.hooks.link_storage(self.storage)
    
    def build_transforms(self):
         return Composer()
     
    def build_hooks(self):
         return InferenceHookList([InferenceProgressBarHook([InferenceFoldProgress(), InferenceStepProgress()])])
            
    @property
    def result(self):
        pass
    
class EnsembleEngine(IEngine):
        def __init__(self, inference_engines:List[InferenceEngine]) -> None:
            super().__init__()
            self.engines = inference_engines
            self.storage = Storage()
        
        def run(self):
            try:
                model_preds = []
                for engine in self.engines:
                    fold_preds = engine.run()
                    model_preds += [fold_preds]
                preds = np.sum(model_preds, axis = 0)
                self.storage.put('ensemble_preds', preds)
            except Exception:
                raise
            
        def build(self, dataloader):
            self.dataloader = dataloader
            for engine in self.engines:
                if self.build_transforms() is not None:
                    engine.build_transforms = self.build_transforms  # type: ignore
                
                if self.build_hooks() is not None:
                    engine.build_hooks = self.build_hooks  # type: ignore
                
                engine.build(self.dataloader)
                
        def build_transforms(self):
            pass     
       
        def build_hooks(self):
            pass
        
        @property
        def result(self):
            return self.storage.get('ensemble_preds')
        