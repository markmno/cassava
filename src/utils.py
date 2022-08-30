import torch
import math
    
class AverageMeter:
    """
    _summary_
    """
    def __init__(self, name:str) -> None:
        """
        _summary_

        Args:
            name (str): _description_
            format (str): _description_
        """
        self.name = name
        self.reset()
        
    def reset(self)->None:
       """
       _summary_
       """
       self.val:float = 0
       self._avg:float = 0
       self.sum:float = 0 
       self.count:int = 0
    
    def update(self, val:float, n:int = 1)->None:
        """
        _summary_

        Args:
            val (float): _description_
            n (int, optional): _description_. Defaults to 1.
        """
        self.val = val
        self.sum += val*n
        self.count += n
        self._avg = self.sum/self.count
        
    def __str__(self):
        return {self.name: self._avg}
     
    @property
    def avg(self):
        return self._avg