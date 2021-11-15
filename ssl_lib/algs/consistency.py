import torch
from .utils import sharpening, tempereture_softmax

class ConsistencyRegularization:
    """
    Basis Consistency Regularization
    Parameters
    --------
    consistency: str
        consistency objective name
    threshold: float
        threshold to make mask
    sharpen: float
        sharpening temperature for target value
    temp_softmax: float
        temperature for temperature softmax
    """
    def __init__(
        self,
        consistency,
        threshold: float = None,
        sharpen: float = None,
        temp_softmax: float = None
    ):
        self.consistency = consistency
        self.threshold = threshold
        self.sharpen = sharpen
        self.tau = temp_softmax

    def __call__(
        self,
        stu_preds,
        tea_logits,
        *args,
        **kwargs
    ):
        mask = self.gen_mask(tea_logits)
        targets = self.adjust_target(tea_logits)
        return stu_preds, targets, mask

    def adjust_target(self, targets):
        if self.sharpen is not None:
            targets = targets.softmax(1)
            targets = sharpening(targets, self.sharpen)
        elif self.tau is not None:
            targets = tempereture_softmax(targets, self.tau)
        else:
            targets = targets.softmax(1)
        return targets
#the returned mask is one if it is not psuodo labeling
    def gen_mask(self, targets):
        targets = targets.softmax(1)
        if self.threshold is None or self.threshold == 0:
            #torch.ones_like: Returns a tensor filled with the scalar value 1, with the same size as input
            return torch.ones_like(targets.max(1)[0])
        #torch.max:Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim
        return (targets.max(1)[0] >= self.threshold).float()

    def __repr__(self):
        return f"Consistency(threshold={self.threshold}, sharpen={self.sharpen}, tau={self.tau})"

    def moving_average(self, parameters):
        ema_factor = min(1 - 1 / (self.global_step + 1), self.ema_factor)
        for emp_p, p in zip(self.model.parameters(), parameters):
            print("-----------------"+emp_p.shape, p.shape)
            emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data