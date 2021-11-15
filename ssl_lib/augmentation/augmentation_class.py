'the code is used from https://github.com/perrying/pytorch-consistency-regularization/blob/master/ssl_lib/augmentation/augmentation_class.py"
import numpy as np
import torch
import torch.nn.functional as F
import random
import torchvision.transforms as tt
from .augmentation_pool import GaussianNoise, GCN, ZCA


class WeakAugmentation:
    """
    Weak augmentation class
    including horizontal flip, random crop, and gaussian noise
    """
    def __init__(
        self,
        img_size: int,
        mean: list,
        scale: list,
        flip=True,
        crop=True,
        noise=True,
        zca=False
    ):
        augmentations = [tt.ToPILImage()]
        if flip:
            augmentations.append(tt.RandomHorizontalFlip())
        if crop:
            augmentations.append(tt.RandomCrop(img_size, int(img_size*0.125), padding_mode="reflect"))
        augmentations += [tt.ToTensor()]
        if zca:
            augmentations += [GCN(), ZCA(mean, scale)]
        else:
            augmentations += [tt.Normalize(mean, scale, True)]
        if noise:
            augmentations.append(GaussianNoise())
        self.augmentations = tt.Compose(augmentations)

    def __call__(self, img):
        return self.augmentations(img)

    def __repr__(self):
        return repr(self.augmentation)
