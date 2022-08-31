import os
from dataclasses import dataclass
from typing import Any, NamedTuple

import pandas as pd
import torch
import torch.optim as optim
from rich import print
from sklearn.model_selection import StratifiedKFold

from src.dataloader import LoadData
from src.engine.train_engine import TrainEngine
from src.hooks.base import TrainHookList
from src.hooks.checkpointer import Checkpointer, EarlyStoppingHook
from src.loss_fn import BiTemperedLogisticLoss
from src.model import TimmModel
from src.utils import seed_everything


@dataclass
class CFG:
    model_name: str = "resnext101_32x8d"
    num_epochs: int = 10
    num_folds: int = 5
    seed: int = 27
    train_img_root: str = "./data/train_images"
    train_df: str = "./data/train.csv"
    weights_root: str = "./weights/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr: float = 1e-4
    min_lr: float = 1e-7
    img_size: int = 512
    train_bs: int = 16
    valid_bs: int = 32
    smoothing: float = 0.2
    num_workers: int = 4
    t1: float = 0.8
    t2: float = 1.4


def run(cfg: CFG):
    seed_everything(cfg.seed)

    train_full_df = pd.read_csv(cfg.train_df)

    train_df_ = train_full_df.copy()

    loader = LoadData(df=train_df_, root=cfg.train_img_root, img_size=cfg.img_size)

    skf = StratifiedKFold(n_splits=cfg.num_folds, random_state=cfg.seed, shuffle=True)
    folds = enumerate(skf.split(train_df_, train_df_["label"]))

    for fold_index, (tr_index, val_index) in folds:
        loss_fn = BiTemperedLogisticLoss(
            t1=cfg.t1, t2=cfg.t2, label_smoothing=cfg.smoothing
        )

        model = TimmModel(model_name=cfg.model_name, pretrained=True)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=cfg.min_lr, last_epoch=-1
        )

        class MyEngine(TrainEngine):
            def build_hooks(self) -> TrainHookList:
                ret = super().build_hooks()
                ret.append(
                    EarlyStoppingHook(
                        patience=11,
                        checkpointer=Checkpointer(
                            self.model,
                            root=os.path.join(cfg.weights_root, cfg.model_name),
                            prefix=f"{fold_index}_fold_",
                        ),
                    )
                )
                return ret

        engine = MyEngine(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=cfg.device,
        )

        print(f"[bold]Start training [{fold_index}/{cfg.num_folds-1}] fold")

        train_loader = loader.get_dataloader(tr_index, cfg.train_bs)

        test_loader = loader.get_dataloader(val_index, cfg.valid_bs)

        class TrainTestTuple(NamedTuple):
            train: Any = train_loader
            eval: Any = test_loader

        engine.build(TrainTestTuple())
        engine.run(cfg.num_epochs)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    run(cfg=CFG())
