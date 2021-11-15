import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(y, target, mask=None):
    if target.ndim == 1: # for hard label
        #because the losses are averaged per element in the batch, we have tensor of losses with size [batch_size].
        # If  reduction="none", we get back theses losses per element in the batch, but by default (reduction="mean") the mean of these losses is returned.
        loss = F.cross_entropy(y, target, reduction="none")
    else:
        #log_softmax: Applies a softmax followed by a logarithm. Parameters (input,
        # dim: A dimension along which log_softmax will be computed)
        # they calculate the crros entropy for a batch ( applying the log max and the sumation)
        loss = -(target * F.log_softmax(y, 1)).sum(1)
    if mask is not None:
        loss = mask * loss
    return loss.mean()

class CrossEntropy(nn.Module):
    def forward(self, y, target, mask=None, *args, **kwargs):
        return cross_entropy(y, target.detach(), mask)