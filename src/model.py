from typing import List, Literal, Tuple
import torch.nn as nn
import timm

model_list:Tuple[str] = tuple(timm.list_models())

class TimmModel(nn.Module):
    def __init__(self, model_name = 'tf_efficientnet_b5_ns',  pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 5)
        
    def forward(self,x):
        return self.model(x)
    

class Effnet(nn.Module):
    def __init__(self, transfer_model = timm.create_model(model_name='tf_efficientnet_b5_ns', 
                                                          pretrained=True)):
        super().__init__()
        self.model = transfer_model
        self.model.classifier = nn.Linear(transfer_model.classifier.in_features, 5)
        
    def forward(self,x):
        return self.model(x)

class ResNext101(nn.Module):
    def __init__(self, model_name:str='resnext101_32x8d', pretrained:bool=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

    def forward(self, x):
        return self.model(x)