import numpy as np
import os
import random
import torch
import torch.backends.cudnn


class AverageMeter:
    """
    Based on https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/utils/avgmeter.html#AverageMeter
    """

    def __init__(self, name: str) -> None:
        """
        Computes and stores the average and current value.

        Args:
            name (str): Name of metric
        
        Examples::
            >>> # Initialize a meter to record loss
            >>> losses = AverageMeter('metric_name')
            >>> # Update meter after every minibatch update
            >>> losses.update(loss_value, batch_size)

        """
        self.name = name
        self.reset()

    def reset(self) -> None:
        """
        Resets all atributes
        """
        self.val: float = 0
        self._avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Updates curent value of metric and compute average

        Args:
            val (float): Value
            n (int, optional): _description_. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    def __str__(self)->dict[str, float]:
        return {self.name: self._avg}

    @property
    def avg(self)->float:
        return round(self._avg, 4)


def seed_everything(seed:int)->None:
    """
    Seeds everything to make experiments more reproducible

    Args:
        seed (int): rangom seed 
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
