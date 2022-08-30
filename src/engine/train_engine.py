from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from rich.traceback import install
from .base import IEngine
from src.utils import AverageMeter
from ..hooks.progressbar_hook import EpochProgress, EvalStepProgress, TrainStepProgress
from ..hooks import TrainHookList, TrainProgressBarHook, LrSchedulerHook
from ..storage import Storage

install(show_locals=False)

__all__ = ["TrainEngine"]

scaler = GradScaler()
device = torch.device("cuda")

class TrainEngine(IEngine):
    """_summary_

    Args:
        IEngine (_type_): _description_
    """
    def __init__(self,
                 model:nn.Module,
                 loss_fn:nn.Module,
                 optimizer:optim.Optimizer,
                 scheduler,
                 eval_func:Optional[Callable[..., torch.Tensor]] = None
                 ) -> None:
        """_summary_

        Args:
            model (nn.Module): _description_
            loss_fn (nn.Module): _description_
            optimizer (optim.Optimizer): _description_
            scheduler (_type_): _description_
            eval_func (Optional[Callable[..., torch.Tensor]], optional): _description_. Defaults to None.
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._eval_func = eval_func if eval_func is not None else loss_fn.to(device) 
   
    def build(self, dataloader):
        self.storage = Storage()
        self.dataloader = dataloader
        self.hooks = self._build_hooks()
        
    def _build_hooks(self)->TrainHookList:
        ret = self.build_hooks()
        ret.append(TrainProgressBarHook([EpochProgress(),TrainStepProgress(),EvalStepProgress()]))
        ret.link_storage(self.storage)
        return ret
        
    def build_hooks(self)->TrainHookList:
        return TrainHookList([LrSchedulerHook(self.scheduler)])
     
    def train_epoch(self)->None:
        self.model.train()
        train_loss = AverageMeter('training_loss') 
        for X_batch, y_batch in self.dataloader.train:
            self.hooks.before_train_step()
            loss = self.train_step(X_batch, y_batch)
            self.hooks.after_train_step()
            loss_val:float = loss.item()
            train_loss.update(loss_val)
            self.storage.put("train_loss", train_loss.avg)

    def eval_epoch(self)->None:
        self.model.eval()
        eval_loss = AverageMeter('eval_loss')
        for X_batch, y_batch in self.dataloader.eval:
            self.hooks.before_eval_step()
            metric = self.eval_step(X_batch, y_batch)
            self.hooks.after_eval_step()
            eval_loss.update(metric.item())
            self.storage.put("metric", eval_loss.avg)
       
    @torch.no_grad()
    def eval_step(self, X_batch, y_batch)->torch.Tensor:
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).long() 
        y_preds = self.model(X_batch)
        return self._eval_func(y_preds, y_batch)
    
    def train_step(self, X_batch:torch.Tensor, y_batch:torch.Tensor):
        X_batch = X_batch.to(device).float()
        y_batch = y_batch.to(device).long()
        with autocast():
            y_preds = self.model(X_batch)
            loss = self.loss_fn(y_preds, y_batch)
        scaler.scale(loss).backward()  # type: ignore
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad()
        return loss
    
    def run(self,
            epochs:int):
        """_summary_

        Args:
            epochs (int): _description_
        """
        self.storage.put('epochs', epochs)
        self.storage.put('train_steps', len(self.dataloader.train))
        self.storage.put('eval_steps', len(self.dataloader.eval))
        try:
            self.hooks.before_run()
            for epoch in range(epochs):
                self.storage.put('epoch', epoch)
                self.hooks.before_epoch()
                self.train_epoch()
                self.eval_epoch()
                self.hooks.after_epoch()
        except Exception:
            raise
        finally:
            self.hooks.after_run()
    
    @property
    def result(self)->None:
        """_summary_
        """
        self.storage.get('best')
