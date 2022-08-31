from typing import Literal
import torch
import torch.nn as nn


class BiTemperedLogisticLoss(nn.Module):
    def __init__(
        self,
        t1: float,
        t2: float,
        label_smoothing: float,
        num_iters: int = 5,
        reduction: Literal["mean", "sum", "none"] = "mean",
    )->None:
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters
        self.reduction = reduction

    @staticmethod
    def tempered_log(x, t) -> torch.Tensor:
        if t == 1.0:
            return torch.log(x)
        else:
            return (x ** (1.0 - t) - 1.0) / (1.0 - t)

    @staticmethod
    def tempered_exp(x, t) -> torch.Tensor:
        if t == 1.0:
            return torch.exp(x)
        else:
            return torch.relu(1.0 + (1.0 - t) * x) ** (1.0 / (1.0 - t))

    def compute_normalization_fixed_point(self, activations) -> torch.Tensor:

        mu = torch.max(activations, dim=-1).values.view(-1, 1)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < self.num_iters:
            i += 1
            logt_partition = torch.sum(
                self.tempered_exp(normalized_activations, self.t2), dim=-1
            ).view(-1, 1)
            normalized_activations = normalized_activations_step_0 * (
                logt_partition ** (1.0 - self.t2)
            )

        logt_partition = torch.sum(
            self.tempered_exp(normalized_activations, self.t2), dim=-1
        ).view(-1, 1)

        return -self.tempered_log(1.0 / logt_partition, self.t2) + mu

    def compute_normalization(self, activations)->torch.Tensor:
        return self.compute_normalization_fixed_point(activations)

    def tempered_softmax(self, activations):
        if self.t2 == 1.0:
            normalization_constants = torch.log(
                torch.sum(torch.exp(activations), dim=-1)
            )
        else:
            normalization_constants = self.compute_normalization(activations)

        return self.tempered_exp(activations - normalization_constants, self.t2)

    def forward(self, activations:torch.Tensor, labels:torch.Tensor):
        if len(labels.shape) < len(activations.shape):  # not one-hot
            labels_onehot = torch.zeros_like(activations)
            labels_onehot.scatter_(1, labels[..., None], 1)
        else:
            labels_onehot = labels

        if self.label_smoothing > 0:
            num_classes = labels_onehot.shape[-1]
            labels_onehot = (
                1 - self.label_smoothing * num_classes / (num_classes - 1)
            ) * labels_onehot + self.label_smoothing / (num_classes - 1)

        temperature1 = self.t1
        temperature2 = self.t2

        probabilities = self.tempered_softmax(activations)
        temp1 = labels_onehot * self.tempered_log(labels_onehot + 1e-10, temperature1)
        temp2 = labels_onehot * self.tempered_log(probabilities, temperature1)
        loss_values = (
            temp1
            - temp2
            - labels_onehot.pow(2.0 - temperature1) / (2.0 - temperature1)
            + probabilities.pow(2.0 - temperature1) / (2.0 - temperature1)
        )
        loss_values = loss_values.sum(dim=-1)  # sum over classes

        if self.reduction == "none":
            return loss_values
        if self.reduction == "sum":
            return loss_values.sum()
        if self.reduction == "mean":
            return loss_values.mean()

    def callback(self):
        return self.loss.backward(retain_graph=True)  # type: ignore
