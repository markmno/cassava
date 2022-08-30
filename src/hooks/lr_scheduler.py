from .base import TrainHook 

class LrSchedulerHook(TrainHook):
    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler
    
    def after_epoch(self):
        self.scheduler.step() 