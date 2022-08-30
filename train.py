from cProfile import label
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple
from src.dataloader import LoadData
from src.hooks.base import TrainHookList
from src.hooks.checkpointer import Checkpointer, EarlyStoppingHook
from src.loss_fn import BiTemperedLogisticLoss
from src.model import Effnet, ResNext101
from src.engine.train_engine import TrainEngine
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import StratifiedKFold
from rich.console import Console
import pandas as pd

@dataclass
class CFG:
    MODEL_NAME: str = 'resnext101_32x8d'
    EPOCHS:int = 10
    NUM_FOLDS:int = 5
    SEED:int = 27
    TRAIN_IMG_ROOT:str = './data/train_images'
    TEST_IMG_ROOT:str = './data/test_images'
    TRAIN_DF:str = './data/train.csv'
    DEVICE:Literal['cuda', 'cpu']= 'cuda'
    NUM_CLASSES:int = 5
    NUM_FOLDS:int = 5
    LR:float = 1e-4
    MIN_LR :float= 1e-7
    IMG_SIZE:int = 512
    TRAIN_BS:int = 16
    VALID_BS:int = 32
    SMOOTHING:float = 0.2
    NUM_WORKERS:int = 4
    t1:float = 0.8
    t2:float = 1.4

def run(cfg:CFG):
    console = Console()
    
    train_full_df = pd.read_csv(cfg.TRAIN_DF) 
     
    train_df_ = train_full_df.copy()
    
    loader = LoadData(df=train_df_, root = cfg.TRAIN_IMG_ROOT, img_size=cfg.IMG_SIZE)
    
    skf = StratifiedKFold(n_splits=CFG.NUM_FOLDS, random_state=CFG.SEED, shuffle=True)
    folds = enumerate(skf.split(train_df_ , train_df_['label']))
    
     
    for fold_index, (tr_index, val_index) in folds:
        loss_fn = BiTemperedLogisticLoss(
                                                             t1 = cfg.t1, 
                                                             t2 = cfg.t2,
                                                             label_smoothing=cfg.SMOOTHING
                                                             )
        
        model = ResNext101(model_name = CFG.MODEL_NAME, pretrained=True)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=CFG.MIN_LR, last_epoch=-1)
        
        class MyEngine(TrainEngine):
            def build_hooks(self) -> TrainHookList:
                ret = super().build_hooks()
                ret.append(EarlyStoppingHook(patience=11, checkpointer=Checkpointer(self.model, f"/home/nmark/projects/cassava/weights/{CFG.MODEL_NAME}/{fold_index}_fold_")))
                return ret
            
        engine = MyEngine(model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)
    
        console.print(f"[bold]Start training [{fold_index}/{CFG.NUM_FOLDS-1}] fold") 
          
        train_loader = loader.get_dataloader(tr_index, cfg.TRAIN_BS) 
        
        test_loader = loader.get_dataloader(val_index, cfg.VALID_BS) 
        
        class TrainTestTuple(NamedTuple):
            train: Any = train_loader
            eval: Any = test_loader
         
        engine.build(TrainTestTuple())    
        engine.run(cfg.EPOCHS)
        
        torch.cuda.empty_cache()

if __name__=='__main__':
    run(cfg=CFG())