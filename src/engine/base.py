from abc import ABC, abstractmethod

__all__ = ["IEngine"]


class IEngine(ABC):
    @abstractmethod
    def build(self):
        """
        Method to add specific arguments before training
        """

    @abstractmethod
    def run(self):
        """
        Method for train/inference to train/predict
        """

    @property
    @abstractmethod
    def result(self):
        """
        Method to get final eval loss/ predictions
        """
