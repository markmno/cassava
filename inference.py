from dataclasses import dataclass, field
from typing import List, NamedTuple
from src import engine
from src.dataloader import TestDataset
from src.engine.inference_engine import InferenceEngine, EnsembleEngine
import torch
import torch.nn as nn
from src.model import Effnet, ResNext101
import pandas as pd
from rich import print
import numpy as np
from scipy.special import softmax
import glob
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.tta import Composer, HFlip, Rotate, VFlip

class Arch(NamedTuple):
    model: nn.Module
    weights: List[str]

@dataclass
class CFG:
    models:List[Arch] = field(default_factory=list)
    device = torch.device('cuda')
    num_workers:int = 8
    image_size:int = 512

device = torch.device('cuda')

def run():
    torch.cuda.empty_cache()
    df = pd.read_csv("/home/nmark/projects/cassava/data/sample_submission.csv")
    
    transforms_valid = A.Compose([
        A.Resize(CFG.image_size, CFG.image_size),
        A.Normalize(),
        ToTensorV2()
    ])
            
    dataset = TestDataset(df=df, root='/home/nmark/projects/cassava/data/test_images', transform=transforms_valid) 
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8)
    
    effnet_engine = InferenceEngine(Effnet(), glob.glob("/home/nmark/projects/cassava/weights/tf_effnet_b5_ns/*.pt")) 
    resnext_engine = InferenceEngine(ResNext101(model_name='resnext101_32x8d'), glob.glob("/home/nmark/projects/cassava/weights/resnext101_32x8d/*.pt"))
    
    class CassavaEnsembleEngine(EnsembleEngine):
        def build_transforms(self):
            return Composer(transforms_list=[VFlip(), HFlip(), Rotate(90), Rotate(180), Rotate(270)])
    
    engine = CassavaEnsembleEngine([effnet_engine, resnext_engine])
    engine.build(test_loader)
    engine.run()
    df['label'] = softmax(engine.result).argmax(1)
    df[['image_id', 'label']].to_csv("./"+'submission.csv', index=False)
    
if __name__ == "__main__":
    run()