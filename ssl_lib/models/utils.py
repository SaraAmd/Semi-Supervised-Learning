import math
import torch.nn as nn
import torch.nn.functional as F
#import resnet3
#form resnet3 import WideResNet
#
# class BaseModel(nn.Module):
#     def forward(self, x):
#         f = self.feature_extractor(x)
#         f = f.mean((2, 3))
#         return self.classifier(f)
#
#     def logits_with_feature(self, x):
#         f = self.feature_extractor(x)
#         f=f.mean((2, 3))
#         c = self.classifier(f)
#         return c, f
#
#     def update_batch_stats(self, flag):
#         # self.modules(), an iterable to the many layers or “modules” defined in the model class
#         for m in self.modules():
#             #isinstance() checks if the particular layer “m” is an instance of a nn.BatchNorm2d layer
#             if isinstance(m, nn.BatchNorm2d):
#                 m.update_batch_stats = flag
#
#
# def conv3x3(i_c, o_c, stride=1, bias=False):
#     return nn.Conv2d(i_c, o_c, 3, stride, 1, bias=bias)
#
# #Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension (Input: (N, C, H, W))
# class BatchNorm2d(nn.BatchNorm2d):
#     def __init__(self, channels, momentum=1e-3, eps=1e-3):
#         super().__init__(channels)
#         self.update_batch_stats = True
#
#     def forward(self, x):
#         if self.update_batch_stats or not self.training:
#             return super().forward(x)
#         else:
#             #nn.functional.batch_norm:Applies Batch Normalization for each channel across a batch of data
#         #torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
#             return nn.functional.batch_norm(
#                 x, None, None, self.weight, self.bias, True, self.momentum, self.eps
#             )
#
#
# def leaky_relu():
#     return nn.LeakyReLU(0.1)

# #the original code comes from
# #https://github.com/perrying/pytorch-consistency-regularization/blob/master/ssl_lib/models/utils.py
# class MT(WideResNet):
#     def __init__(self, model, ema_factor):
#         super().__init__()
#         self.model = model
#         self.model.train()
#         self.ema_factor = ema_factor
#         self.global_step = 0
#
#     # def forward(self, x, y, model, mask):
#     #     self.global_step += 1
#     #     y_hat = self.model(x)
#     #     model.update_batch_stats(False)
#     #     y = model(x) # recompute y since y as input of forward function is detached
#     #     model.update_batch_stats(True)
#     #     return (F.mse_loss(y.softmax(1), y_hat.softmax(1).detach(), reduction="none").mean(1) * mask).mean()
#
#     def moving_average(self, parameters):
#         ema_factor = min(1 - 1 / (self.global_step+1), self.ema_factor)
#         for emp_p, p in zip(self.model.parameters(), parameters):
#             emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data
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


def __buffer_update(ema_model, raw_model, factor):
    """ema for buffer parameters (e.g., running_mean and running_var in nn.BatchNorm2d)"""
    for ema_p, raw_p in zip(ema_model.buffers(), raw_model.buffers()):
        ema_p.data = __ema(ema_p.data, raw_p.data, factor)
    # """copy buffer parameters (e.g., running_mean and running_var in nn.BatchNorm2d)"""
    # for ema_p, raw_p in zip(ema_model.buffers(), raw_model.buffers()):
    #     ema_p.copy_(raw_p)


def ema_update(ema_model, raw_model, ema_factor, weight_decay_factor=None, global_step=None):
    if global_step is not None:
        ema_factor = min(1 - 1 / (global_step+1), ema_factor)
    __param_update(ema_model, raw_model, ema_factor)
    __buffer_update(ema_model, raw_model, ema_factor)
    if weight_decay_factor is not None:
        apply_weight_decay(ema_model.modules(), weight_decay_factor)