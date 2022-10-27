import os
from typing import Optional

import torch
import torch.nn as nn

from .base import TrainHook


class EarlyStoppingError(Exception):
    pass

class Checkpointer(TrainHook):
    def __init__(self, model: nn.Module, root: str, prefix: Optional[str]) -> None:
        """
        Hook that makes model checkpoints.

        Args:
            model (nn.Module): Pytorch Model
            root (str): Path to saved models
            prefix (Optional[str]): Saved checkpoint prefix 
        """
        super().__init__()
        self.model = model
        self.path = self.make_path(root, prefix)

    def make_path(self, root: str, prefix):
        if os.path.exists(root) == False:
            os.mkdir(root)
        return os.path.join(root, prefix) if prefix is not None else root

    def after_epoch(self):
        torch.save(self.model.state_dict(), self.path + "model_best.pt")

    def after_run(self):
        torch.save(self.model.state_dict(), self.path + "model_last.pt")


class EarlyStoppingHook(TrainHook):
    def __init__(self, patience: int, checkpointer: Checkpointer) -> None:
        """
        Early Stopping Hook

        Args:
            patience (int): Num of epochs before early stopping
            checkpointer (Checkpointer): Checkpointer Class
        """
        super().__init__()
        self.patience = patience
        self.checkpointer = checkpointer

    def before_run(self):
        self.counter: int = 0

    def after_epoch(self):
        current = self.storage.get("metric")
        best = self.storage.get("best_metric")

        if best is None:
            self.storage.put("best_metric", current)
        elif best <= current:
            self.counter += 1
        else:
            self.counter = 0
            self.storage.put("best_metric", current)

        self.storage.put("stopping_counter", self.counter)

        if self.counter > self.patience:
            self.checkpointer.after_epoch()
            raise EarlyStoppingError

    def after_run(self):
        self.checkpointer.after_run()
