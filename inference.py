import glob
import os
from dataclasses import dataclass

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from scipy.special import softmax
from torch.utils.data import DataLoader

import src.tta as tta
from src.dataloader import TestDataset
from src.engine.inference_engine import EnsembleEngine, InferenceEngine
from src.model import TimmModel
from src.utils import seed_everything


@dataclass
class CFG:
    models_names = ["tf_efficientnet_b5_ns", "resnext101_32x8d"]
    device = torch.device("cuda")
    test_image_root: str = "./data/test_images/"
    csv_path: str = "./data/sample_submission.csv"
    weights_path: str = "./weights/"
    num_workers: int = 8
    image_size: int = 512
    seed: int = 27


def run(cfg: CFG):
    seed_everything(cfg.seed)
    df = pd.read_csv(cfg.csv_path)

    transforms_valid = A.Compose(
        [A.Resize(cfg.image_size, cfg.image_size), A.Normalize(), ToTensorV2()]
    )

    dataset = TestDataset(df=df, root=cfg.test_image_root, transform=transforms_valid)
    test_loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=8
    )

    engine_list = [
        InferenceEngine(
            TimmModel(model_name),
            glob.glob(os.path.join(CFG.weights_path, model_name, "*.pt")),
            device=cfg.device,
        )
        for model_name in cfg.models_names
    ]

    class CassavaEnsembleEngine(EnsembleEngine):
        def build_transforms(self):
            return tta.Composer(
                transforms_list=[
                    tta.VFlip(),
                    tta.HFlip(),
                    tta.Rotate(90),
                    tta.Rotate(180),
                    tta.Rotate(270),
                ]
            )

    engine = CassavaEnsembleEngine(engine_list)
    engine.build(test_loader)
    engine.run()
    df["label"] = softmax(engine.result).argmax(1)
    df[["image_id", "label"]].to_csv("./" + "submission.csv", index=False)


if __name__ == "__main__":
    run(cfg=CFG())
