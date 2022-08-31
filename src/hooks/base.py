from typing import List

import torch

from ..storage import Storage

device = torch.device("cuda")

__all__ = ["TrainHook", "InferenceHook", "InferenceHookList", "TrainHookList"]


class IHook:
    storage: "Storage"

    def before_run(self):
        pass

    def after_run(self):
        pass


class TrainHook(IHook):
    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_train_step(self):
        pass

    def after_train_step(self):
        pass

    def before_eval_step(self):
        pass

    def after_eval_step(self):
        pass


class TrainHookList(TrainHook):
    def __init__(self, hook_list: List[TrainHook]):
        self.hook_list = hook_list

    def append(self, value):
        self.hook_list.append(value)

    def link_storage(self, storage: Storage):
        for hook in self.hook_list:
            hook.__dict__["storage"] = storage

    def before_run(self):
        for hook in self.hook_list:
            hook.before_run()

    def after_run(self):
        for hook in self.hook_list:
            hook.after_run()

    def before_epoch(self):
        for hook in self.hook_list:
            hook.before_epoch()

    def after_epoch(self):
        for hook in self.hook_list:
            hook.after_epoch()

    def before_train_step(self):
        for hook in self.hook_list:
            hook.before_train_step()

    def after_train_step(self):
        for hook in self.hook_list:
            hook.after_train_step()

    def before_eval_step(self):
        for hook in self.hook_list:
            hook.before_eval_step()

    def after_eval_step(self):
        for hook in self.hook_list:
            hook.after_eval_step()


class InferenceHook(IHook):
    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_inference_step(self):
        pass

    def after_inference_step(self):
        pass

    def before_eval_step(self):
        pass

    def after_eval_step(self):
        pass


class InferenceHookList(InferenceHook):
    def __init__(self, hook_list: List[InferenceHook]):
        self.hook_list = hook_list

    def append(self, value):
        self.hook_list.append(value)

    def link_storage(self, storage: Storage):
        for hook in self.hook_list:
            hook.__dict__["storage"] = storage

    def before_run(self):
        for hook in self.hook_list:
            hook.before_run()

    def after_run(self):
        for hook in self.hook_list:
            hook.after_run()

    def before_epoch(self):
        for hook in self.hook_list:
            hook.before_epoch()

    def after_epoch(self):
        for hook in self.hook_list:
            hook.after_epoch()

    def before_inference_step(self):
        for hook in self.hook_list:
            hook.before_inference_step()

    def after_inference_step(self):
        for hook in self.hook_list:
            hook.after_inference_step()

    def before_eval_step(self):
        for hook in self.hook_list:
            hook.before_eval_step()

    def after_eval_step(self):
        for hook in self.hook_list:
            hook.after_eval_step()
