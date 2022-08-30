from abc import ABC, abstractmethod

__all__ = ["IEngine"]


class IEngine(ABC):
    @abstractmethod
    def build(self):
        """_summary_
        """
    
    @abstractmethod
    def run(self):
        """
        Method for train/inference to train/predict
        
        Example:
        >>> engine = Engine()
        >>> engine.run()
        """
    
    @property
    @abstractmethod
    def result(self):
        #TODO: write docs
        """
        
        """
    