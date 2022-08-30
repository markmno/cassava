from ..storage import Storage
from .base import TrainHookList, TrainHook, InferenceHook, InferenceHookList
from rich.panel import Panel
from rich.console import Group
from rich.progress import Progress, ProgressColumn, TextColumn, SpinnerColumn
from rich.text import Text
from rich.live import Live

class MetricsTextColumn(ProgressColumn):
        """A column containing text."""

        def __init__(self):
            self._tasks = {}
            self._current_task_id = 0
            self._metrics = {}
            super().__init__()

        def update(self, metrics):
            # Called when metrics are ready to be rendered.
            # This is to prevent render from causing deadlock issues by requesting metrics
            # in separate threads.
            self._metrics = metrics

        def render(self, task) -> Text:
            if task.id not in self._tasks:
                self._tasks[task.id] = "None"
                if self._renderable_cache:
                    self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]   # type: ignore
                self._current_task_id = task.id
            if task.id != self._current_task_id:
                return self._tasks[task.id]

            text = ""
            for k, v in self._metrics.items():
                text += f"| {k}: {v} "
            return Text(text, justify="left")
        

class MyProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            additional_columns = task.fields.get("additional_columns")
            if additional_columns is not None:
                self.columns = (*self.columns, *additional_columns)
            yield self.make_tasks_table([task])


class TrainProgressBarHook(TrainHookList):
    def __init__(self, bar_list:list):
        super().__init__(bar_list)
        self.bar_list = bar_list
        self.live = Live(self.get_progress())
   
    def link_storage(self, storage: Storage):
        return super().link_storage(storage)
    
    def get_progress(self):
        bars = []
        for bar in self.bar_list:
            bars.append(bar.get_progress())
        return Panel(Group(*bars), expand=False, safe_box=True, highlight=True)    
    
    def before_run(self):
        self.link_storage(self.storage)
        self.live.start()
        super().before_run()
    
    def before_epoch(self):
        for bar in self.bar_list:    
            bar.before_epoch() 
    
    def before_train_step(self):
        for bar in self.bar_list:    
            bar.before_train_step() 
    
    def after_train_step(self):
        for bar in self.bar_list:    
            bar.after_train_step() 
    
    def before_eval_step(self):
        for bar in self.bar_list:    
            bar.before_eval_step() 
    
    def after_eval_step(self):
        for bar in self.bar_list:    
            bar.after_eval_step() 
    
    def after_epoch(self):
        for bar in self.bar_list:    
            bar.after_epoch() 
    
    def after_run(self):
        self.live.stop()

class EpochProgress(TrainHook):
    def __init__(self)->None:
        self.get_progress()
 
    def get_progress(self):
        self._best_component = MetricsTextColumn()
        self._stopping_counter = MetricsTextColumn()
        self.epoch_progress = Progress(SpinnerColumn(spinner_name="arrow3"),
                                                 TextColumn('epoch [{task.completed}/[cyan]{task.total}]'),
                                                 self._best_component, 
                                                 self._stopping_counter          
                                                 )
        return self.epoch_progress

    def before_run(self) -> None:
        self.epoch = self.epoch_progress.add_task("[purple]epoch", total=self.storage.get('epochs'))

    def after_epoch(self) -> None:
        self._best_component.update(self.storage.best_metric)
        self._stopping_counter.update({"stopping counter":self.storage.get("stopping_counter")})
        self.epoch_progress.advance(self.epoch)

class TrainStepProgress(TrainHook):
    def __init__(self)->None:
        self.get_progress()
    
    def get_progress(self):
        self._train_loss_component = MetricsTextColumn()
        self.train_progress = Progress(
                                        # SpinnerColumn(),
                                        *Progress.get_default_columns(),
                                        self._train_loss_component,
                                      )
        return self.train_progress
     
    def before_epoch(self) -> None:
        self.train_step = self.train_progress.add_task("[red]train", total=self.storage.get('train_steps'), additional_columns = [self._train_loss_component])

    def after_train_step(self) -> None:
        train_loss = self.storage.train_loss
        self._train_loss_component.update(train_loss)        
        self.train_progress.update(self.train_step, advance=1)

    def after_epoch(self) -> None:
        self.train_progress.remove_task(self.train_step)
         
class EvalStepProgress(TrainHook):
    def __init__(self)->None:
        self.get_progress()

    def get_progress(self):
        self._metric_component = MetricsTextColumn()
        self.eval_progress = Progress(
                                        # SpinnerColumn(),
                                        *Progress.get_default_columns(),
                                       self._metric_component
                                      )
        return self.eval_progress

    def before_epoch(self) -> None:
        self.eval_step = self.eval_progress.add_task("[blue]eval ", total=self.storage.get('eval_steps'), additional_columns = [self._metric_component])

    def after_eval_step(self)->None:
        self._metric_component.update(self.storage.metric)
        self.eval_progress.update(self.eval_step, advance=1)

    def after_epoch(self) -> None:
        self.eval_progress.remove_task(self.eval_step)
        
        
class InferenceProgressBarHook(InferenceHookList):
    def __init__(self, bar_list:list):
        super().__init__(bar_list)
        self.bar_list = bar_list
        self.live = Live(self.get_progress())
   
    def link_storage(self, storage: Storage):
        return super().link_storage(storage)
    
    def get_progress(self):
        bars = []
        for bar in self.bar_list:
            bars.append(bar.get_progress())
        return Panel(Group(*bars), expand=False, safe_box=True, highlight=True)    
    
    def before_run(self):
        self.link_storage(self.storage)
        self.live.start()
        super().before_run()
    
    def after_run(self):
        super().after_run()
        self.live.stop()
    
class InferenceStepProgress(InferenceHook):
    def __init__(self) -> None:
        super().__init__()
        self.get_progress()
        
    def get_progress(self):
        # self._metric_component = MetricsTextColumn()
        self.eval_progress = Progress(
                                        # SpinnerColumn(),
                                        *Progress.get_default_columns(),
                                    #    self._metric_component
                                      )
        return self.eval_progress

    def before_epoch(self) -> None:
        self.eval_step = self.eval_progress.add_task("[blue]infer", total=self.storage.get('eval_steps')) 

    def after_eval_step(self)->None:
        # self._metric_component.update(self.storage.metric)
        self.eval_progress.update(self.eval_step, advance=1)

    def after_epoch(self) -> None:
        self.eval_progress.remove_task(self.eval_step)

class InferenceFoldProgress(InferenceHook):
    def __init__(self)->None:
        super().__init__()
        self.get_progress()
    
    def get_progress(self):
        self.epoch_progress = Progress(SpinnerColumn(spinner_name="arrow3"),
                                                 TextColumn('fold [{task.completed}/[cyan]{task.total}]'),
                                                 )
        return self.epoch_progress

    def before_run(self) -> None:
        self.epoch = self.epoch_progress.add_task("[purple]fold", total=self.storage.get('folds'))

    def after_epoch(self) -> None:
        self.epoch_progress.advance(self.epoch)
