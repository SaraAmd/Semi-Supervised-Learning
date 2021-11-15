
import torch
import numpy as np

"""
For torch.Tensor
"""
class GaussianNoise:
    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, x):
        with torch.no_grad():
            return x + torch.randn_like(x) * self.std

    def __repr__(self):
        return f"GaussianNoise(std={self.std})"

class ZCA:
    def __init__(self, mean, scale):
        self.mean = torch.from_numpy(mean).float()
        self.scale = torch.from_numpy(scale).float()

    def __call__(self, x):
        c, h, w = x.shape
        x = x.reshape(-1)
        x = (x - self.mean) @ self.scale
        return x.reshape(c, h, w)

    def __repr__(self):
        return f"ZCA()"


class GCN:
    """global contrast normalization"""
    def __init__(self, multiplier=55, eps=1e-10):
        self.multiplier = multiplier
        self.eps = eps

    def __call__(self, x):
        x -= x.mean()
        norm = x.norm(2)
        norm[norm < self.eps] = 1
        return self.multiplier * x / norm

    def __repr__(self):
        return f"GCN(multiplier={self.multiplier}, eps={self.eps})"


"""
For numpy.array
"""
def numpy_batch_gcn(images, multiplier=55, eps=1e-10):
    # global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    return multiplier * images / per_image_norm

