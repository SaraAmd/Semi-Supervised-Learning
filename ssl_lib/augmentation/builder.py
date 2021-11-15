
from .augmentation_class import WeakAugmentation


def gen_weak_augmentation(img_size, mean, std, flip=True, crop=True, noise=True, zca=False):
    return WeakAugmentation(img_size, mean, std, flip, crop, noise, zca)