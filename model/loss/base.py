
import torch
import torch.nn as nn


class BaseLoss(nn.Module):

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.last_components = {}

    def auxiliary(self, aux: tuple, logits: torch.Tensor, target: torch.Tensor):
        return logits.new_zeros(()), {}

    def forward(self, input, target):
        if not isinstance(input, (tuple, list)):
            ce = self.ce_loss(input, target)
            self.last_components = {"ce": (ce.item(), 1.0)}
            return ce

        logits, aux = input[0], tuple(input[1:])
        ce = self.ce_loss(logits, target)

        extra, components = self.auxiliary(aux, logits, target)
        self.last_components = {"ce": (ce.item(), 1.0), **components}

        return ce + extra
