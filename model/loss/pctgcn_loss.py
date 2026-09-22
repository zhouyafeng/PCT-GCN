from .base import BaseLoss


class CrossEntropyOrthogonalLoss(BaseLoss):

    def __init__(self, orth_weight: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.orth_weight = orth_weight

    def auxiliary(self, aux, logits, target):
        orth = aux[0].mean()
        extra = self.orth_weight * orth
        return extra, {"orth": (orth.item(), self.orth_weight)}
