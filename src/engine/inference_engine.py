from typing import List, Literal

import numpy as np
import torch
import torch.nn as nn

from ..hooks import InferenceHookList, InferenceProgressBarHook
from ..hooks.progressbar_hook import InferenceFoldProgress, InferenceStepProgress
from ..storage import Storage
from ..tta import Composer
from .base import IEngine

__all__ = ["InferenceEngine", "EnsembleEngine"]


class InferenceEngine(IEngine):
    def __init__(
        self, model: nn.Module, weights: List[str] | str, device: torch.device
    ) -> None:
        """
        Runs inference on your model with given weights

        Args:
            model (nn.Module): Pytorch Model
            weights (List[str] | str): List of paths to your model weights
            device (torch.device): torch device (GPU/CPU/TPU)
        
        Example:
        >>> engine = InferenceEngine()
        >>> engine.build(dataloader)
        >>> engine.run()
        """
        super().__init__()
        model.eval()
        self.device = device
        self.model = model.to(self.device)
        self.weights_list = weights

    def run(self) -> np.ndarray:
        self.storage.put("eval_steps", len(self.dataloader))
        self.storage.put("folds", len(self.weights_list))
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
            image = image.to(self.device)
            aug_images = self.augment(image)
            tta_preds = []
            for aug_image in aug_images:
                logits = self.model(aug_image)
                tta_preds += [torch.softmax(logits, 1)]
            self.hooks.after_inference_step()
            preds += [torch.cat(tta_preds, 0).mean(0, keepdim=True)]
        return torch.cat(preds, dim=0).detach().cpu().numpy()

    def augment(self, image: torch.Tensor):
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
        return InferenceHookList(
            [
                InferenceProgressBarHook(
                    [InferenceFoldProgress(), InferenceStepProgress()]
                )
            ]
        )

    @property
    def result(self):
        pass


class EnsembleEngine(IEngine):
    def __init__(
        self,
        inference_engines: List[InferenceEngine],
        reduction: Literal["mean", "sum"] = "sum",
    ) -> None:
        """_summary_

        Args:
            inference_engines (List[InferenceEngine]): _description_
            reduction (Literal[&#39;mean&#39;, &#39;sum&#39;], optional): _description_. Defaults to 'sum'.
        """
        super().__init__()
        self.engines = inference_engines
        self.reduction_func = self._build_reduction_func(reduction)
        self.storage = Storage()

    def run(self):
        try:
            model_preds = []
            for engine in self.engines:
                fold_preds = engine.run()
                model_preds += [fold_preds]
            preds = self.reduction_func(model_preds, axis=0)
            self.storage.put("ensemble_preds", preds)
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

    def _build_reduction_func(self, reduction):
        if reduction == "mean":
            return np.mean
        return np.sum

    @property
    def result(self):
        return self.storage.get("ensemble_preds")
