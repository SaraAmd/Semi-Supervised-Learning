'''
The dataset is comprised of 60,000 32×32 pixel color photographs of objects from 10 classes
The class labels and their standard associated integer values are listed below.
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
after reordering:
0:bird, 1:cat, 2:deer, 3:dog, 4:frog, 5:horse, 6:ship, 7:truk, 8:plain, 9, automobile
'''

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Sampler
from torch.utils.data import Dataset
import torch.utils.data
import numpy as np


########################################################################
#we need to make a dataset class because we will use it for the sampler class that thae takes a dataset object

#adopted from https://github.com/perrying/pytorch-consistency-regularization/blob/6624c4e0bb1813b5952445ce34f9d4e52484ce38/ssl_lib/datasets/dataset_class.py#L4
class LabeledDataset:
    """
    For labeled dataset
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image = torch.from_numpy(self.dataset["images"][idx]).float()
        image = image.permute(2, 0, 1).contiguous() / 255.
        label = int(self.dataset["labels"][idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset["images"])


    def __len__(self):
        return len(self.dataset["images"])

class UnlabeledDataset:
    """
    For unlabeled dataset
    """
    def __init__(self, dataset, weak_augmentation=None, strong_augmentation=None):
        self.dataset = dataset
        self.weak_augmentation = weak_augmentation
        self.strong_augmentation = strong_augmentation

    def __getitem__(self, idx):
        image = torch.from_numpy(self.dataset["images"][idx]).float()
        image = image.permute(2, 0, 1).contiguous() / 255.
        label = int(self.dataset["labels"][idx])
        w_aug_image = self.weak_augmentation(image)
        if self.strong_augmentation is not None:
            s_aug_image = self.strong_augmentation(image)
        else:
            s_aug_image = self.weak_augmentation(image)
        return w_aug_image, s_aug_image, label

    def __len__(self):
        return len(self.dataset["images"])


 #adopted from DSL3
# n_labels= we need 2400 labeled samples i.e 400 samples per 6-class(n_labels_per_cls= n_labels/6)
# n_unlabels= we randomly select n_unlabelde (20000) samples from the th 10 calsses
#tot_class = is the total number of classes that we want to classify from all the classes available in dataset
#ratio= is the ratio of unlabeled images from the other four classes to modulate class distribution mismatch (50%
# means half of the unlabeled data comes from  the 6 classes (animal classes))(In distributioan samples) and
# the others come from OOD samples (the four classes( the viecle classes for CIFAR10).
# 0 % mean all  unlabeled data comes from  the In Distribution (ID) samples (the six class( animal class) for CIFAR10)
# and there is no OOD samples.
#https://github.com/perrying/realistic-ssl-evaluation-pytorch/blob/master/build_dataset.py
#this function returns a dictionary of labled train daataset and unlabeled dataset
'''
this function split train dataset into labeled and unlabeled
@:param:train_set: it takes a dictionary of the data where the keys are the "images" and and  "lables"
and the values for the key is a list of data and their labels accordingly

@:return: a dictonary for the lables trainset and a dictionary for the unlabled dataset
'''
rng=np.random.RandomState(96)
def split_l_u(cfg,train_set, n_labels, n_unlabels, n_class, ratio ):
    # NOTE: this function assume that train_set is shuffled.

    images = train_set["images"]
    labels = train_set["labels"]
    # number of total classes in the input dataset (dataset is a dictionary)
    classes = np.unique(labels)
    # numeber of samples in each id class in unlabelde dataset
    n_labels_per_cls = n_labels // n_class
    #adopted from DSL3
    #consider varying mismatch ratio
    n_unlabels_per_cls = int(n_unlabels*(1.0-ratio)) // n_class
    if(n_class < len(classes)):
        n_unlabels_shift = (n_unlabels - (n_unlabels_per_cls * n_class)) // (len(classes) - n_class)
        print("n_unlabels_shift is ", n_unlabels_shift)
    #............
    l_images = []
    l_labels = []
    u_images = []
    u_labels = []
    for c in classes[:n_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:n_labels_per_cls]]
        l_labels += [c_labels[:n_labels_per_cls]]
        u_images += [c_images[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
        u_labels += [c_labels[n_labels_per_cls:n_labels_per_cls+n_unlabels_per_cls]]
    #adopted from DSL3
    #In this section of the code they want to include mismatch classes into the unlabeled data
    #adding equal samples from all OOD classes that are available(in case of cifar ther are 4 clasees in OOD)
    for c in classes[n_class:]:
        cls_mask = (labels == c)
        print("cls_mask", cls_mask)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        u_images += [c_images[:n_unlabels_shift]]
        u_labels += [c_labels[:n_unlabels_shift]]
    #............
    l_train_set = {"images": np.concatenate(l_images, 0), "labels": np.concatenate(l_labels, 0)}
    u_train_set = {"images": np.concatenate(u_images, 0), "labels": np.concatenate(u_labels, 0)}


    # permute index of training set for labeled dataset
    indices_l_train_set = rng.permutation(len(l_train_set["images"]))

    l_train_set["images"] = l_train_set["images"][indices_l_train_set]
    l_train_set["labels"] = l_train_set["labels"][indices_l_train_set]


    # permute index of training set for unlabeled data
    indices_u_train_set = rng.permutation(len(u_train_set["images"]))
    u_train_set["images"] = u_train_set["images"][indices_u_train_set]
    u_train_set["labels"] = u_train_set["labels"][indices_u_train_set]

    #...............
    return l_train_set, u_train_set, indices_l_train_set, indices_u_train_set
#split test dataset for getting only the 6 classes that we want to classify for test dataset and validaton dataset
#returns a dictonary of the dataset
def split_test(cfg,test_set, n_class):

    images = test_set["images"]
    labels = test_set['labels']
    classes = np.unique(labels)
    l_images = []
    l_labels = []
    for c in classes[:n_class]:
        cls_mask = (labels == c)
        c_images = images[cls_mask]
        c_labels = labels[cls_mask]
        l_images += [c_images[:]]
        l_labels += [c_labels[:]]
    test_set = {"images": np.concatenate(l_images, 0), "labels":np.concatenate(l_labels,0)}
    # permute index of test set for
    indices = rng.permutation(len(test_set["images"]))
    test_set["images"] = test_set["images"][indices]
    test_set["labels"] = test_set["labels"][indices]
    return test_set

#Loading train and test data set from torchvision
#CIFAR10 images is stored as regular numpy array and the dtype is uint8, the shape of train
# samples id (50000, 32,32 3), the targets are stored as list
#returns two dictionaries, one for test and the other for train

def load_cifar10(root):

    train_data = torchvision.datasets.CIFAR10(root, download=True)
    test_data = torchvision.datasets.CIFAR10(root, train=False)
    train_data = {"images": train_data.data.astype(np.float32),
                  "labels": np.asarray(train_data.targets).astype(np.int32)}
    test_data = {"images": test_data.data.astype(np.float32),
                 "labels": np.asarray(test_data.targets).astype(np.int32)}
    return train_data, test_data



def get_zca_normalization_param(images, scale=0.1, eps=1e-10):
    n_data, height, width, channels = images.shape

    images = images.transpose(0, 3, 1, 2)
    images = images.reshape(n_data, height*width*channels)
    image_cov = np.cov(images, rowvar=False)
    U, S, _ = np.linalg.svd(image_cov + scale * np.eye(image_cov.shape[0]))
    zca_decomp = np.dot(U, np.dot(np.diag(1/np.sqrt(S + eps)), U.T))
    mean = images.mean(axis=0)
    return mean, zca_decomp

# class ZCA:
#     def __init__(self, mean, scale):
#         self.mean = torch.from_numpy(mean).float()
#         self.scale = torch.from_numpy(scale).float()
#
#     def __call__(self, x):
#         c, h, w = x.shape
#         x = x.reshape(-1)
#         x = (x - self.mean) @ self.scale
#         return x.reshape(c, h, w)
#
#     def __repr__(self):
#         return f"ZCA()"
#
#
# class GCN:
#     """global contrast normalization"""
#     def __init__(self, multiplier=55, eps=1e-10):
#         self.multiplier = multiplier
#         self.eps = eps
#
#     def __call__(self, x):
#         x -= x.mean()
#         norm = x.norm(2)
#         norm[norm < self.eps] = 1
#         return self.multiplier * x / norm
#
#     def __repr__(self):
#         return f"GCN(multiplier={self.multiplier}, eps={self.eps})"
#
#
# """
# For numpy.array
# """
# def numpy_batch_gcn(images, multiplier=55, eps=1e-10):
#     # global contrast normalization
#     images = images.astype(np.float)
#     images -= images.mean(axis=(1,2,3), keepdims=True)
#     per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
#     per_image_norm[per_image_norm < eps] = 1
#     return multiplier * images / per_image_norm
#
#





    
