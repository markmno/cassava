from dataclasses import dataclass, field
from importlib.metadata import files
from pickle import TRUE
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Storage"]

def storage_factory(cfg=None):
    """
    Parse config file and create Storage object which have reserved slots for nessesary scalar values
    """
    
    slots:List[str] = ['metric', 'train_loss', 'best_metric']
    if cfg is not None:
        slots:List[str] = cfg.parse_slots()
    
    @dataclass(slots=True)
    class ValueDict:
        results:Dict[str, float|int ] =  field(default_factory=dict)
        def __post_init__(self):
            for slot in slots:
                object.__setattr__(ValueDict, slot, None)
    
    return Storage(ValueDict())

@dataclass
class ValueDict:
    metric:Optional[float] = None
    train_loss:Optional[float] = None
    best_metric:float = np.Inf
    
    def __getitem__(self, name:str)->None:
        return self.__dict__[name]
    
    def __setitem__(self, name:str, value:Any)->None:
        self.__dict__[name] = value
        
class Storage:
    """_summary_
    """
    def __init__(self, value_dict=None) -> None:
        """_summary_

        Args:
            value_dict (_type_, optional): _description_. Defaults to None.
        """
        self._dict = value_dict if value_dict is not None else ValueDict()
        
    def accumulate(self, name:str, value: Any)->None:
        """_summary_

        Args:
            name (str): _description_
            value (Any): _description_
        """
        try:
            self._dict[name] += [value]  # type: ignore
        except:
            self._dict[name] = [value]
        
    def clean(self, name:str)->None:
        """_summary_

        Args:
            name (str): _description_
        """
        self._dict[name] = None
    
    def put(self, name:str, value: Any)->None:
        """_summary_

        Args:
            name (str): _description_
            value (Any): _description_
        """
        self._dict[name] = value
        
    def get(self, name:str) -> Any:
        """_summary_

        Args:
            name (str): _description_

        Returns:
            Any: _description_
        """
        # if isinstance(self._dict[name], list):
        #     return np.mean(self._dict[name])
        return self._dict[name]
    
    def find_best(self, name:str) -> Any:
        """_summary_

        Args:
            name (str): _description_

        Returns:
            Any: _description_
        """
        pass
     
    def __repr__(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return self._dict.__repr__()
    
    @property
    def metric(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {"metric": self._dict.metric}
    
    @property
    def train_loss(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {"loss": self._dict.train_loss}
    
    @property
    def best_metric(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return {"best": self._dict.best_metric}