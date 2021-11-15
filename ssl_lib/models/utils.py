import math
import torch.nn as nn
import torch.nn.functional as F

"""
For exponential moving average
"""

def apply_weight_decay(modules, decay_rate):
    """apply weight decay to weight parameters in nn.Conv2d and nn.Linear"""
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data -= decay_rate * m.weight.data


def param_init(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            f, _, k, _ = m.weight.shape
    #torch.nn.init.normal_(tensor, mean=0.0, std=1.0):Fills the input Tensor with values drawn from the normal distribution
            nn.init.normal_(m.weight, 0, 1./math.sqrt(0.5 * k * k * f))
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


def __ema(p1, p2, factor):
    return factor * p1 + (1 - factor) * p2


def __param_update(ema_model, raw_model, factor):
    """ema for trainable parameters"""
    for ema_p, raw_p in zip(ema_model.parameters(), raw_model.parameters()):
        ema_p.data = __ema(ema_p.data, raw_p.data, factor)
        raw_p.data.copy_(ema_p.data)



