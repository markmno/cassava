from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["Storage"]


def storage_factory(cfg=None):
    """
    Parse config file and create Storage object which have reserved slots for nessesary scalar values
    """
    slots: List[str] = ["metric", "train_loss", "best_metric"]
    if cfg is not None:
        slots: List[str] = cfg.parse_slots()

    @dataclass(slots=True)
    class ValueDict:
        results: Dict[str, float | int] = field(default_factory=dict)

        def __post_init__(self):
            for slot in slots:
                object.__setattr__(ValueDict, slot, None)

    return Storage(ValueDict())


@dataclass
class ValueDict:
    metric: Optional[float] = None
    train_loss: Optional[float] = None
    best_metric: float = np.Inf

    def __getitem__(self, name: str) -> None:
        return self.__dict__[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value


class Storage:
    def __init__(self, value_dict=None) -> None:
        """
        Class to store training/inference data/

        Args:
            value_dict (_type_, optional): Dict of default values. Defaults to None.
        """
        self._dict = value_dict if value_dict is not None else ValueDict()

    def accumulate(self, name: str, value: Any) -> None:
        """_summary_

        Args:
            name (str): _description_
            value (Any): _description_
        """
        try:
            self._dict[name] += [value]  # type: ignore
        except:
            self._dict[name] = [value]

    def clean(self, name: str) -> None:
        """_summary_

        Args:
            name (str): _description_
        """
        self._dict[name] = None

    def put(self, name: str, value: Any) -> None:
        """
        Method to put value in a Storage
        """
        self._dict[name] = value

    def get(self, name: str) -> Any:
        """
        Method to get a value by a name.
        """
        return self._dict[name]

    def __repr__(self) -> str:
        return self._dict.__repr__()

    @property
    def metric(self):
        return {"metric": self._dict.metric}

    @property
    def train_loss(self):
        return {"loss": self._dict.train_loss}

    @property
    def best_metric(self):
        return {"best": self._dict.best_metric}
