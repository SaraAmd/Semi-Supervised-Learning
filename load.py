import numpy as np

import torch.utils.data
from torchvision import transforms
import dataset
from ssl_lib.augmentation.builder import  gen_weak_augmentation
from ssl_lib.augmentation.augmentation_pool import numpy_batch_gcn, ZCA, GCN
import random
_DATA_DIR = "data"
from torch.utils.data import Sampler

'''
all the input arguments are the strings from the arg.parse
@:param dataset is the train from the ars.parse
@:return a dataloader object


'''
class InfiniteSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        epochs = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(epochs)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def get_dataloaders(root,data, n_labels, n_unlabels, n_valid, l_batch_size, ul_batch_size, test_batch_size, iterations,
                    n_class, ratio, unlabeled_aug,cfg, logger=None):

    if data == "CIFAR10":

        train_set, test_set = dataset.load_cifar10(root)

        #move class "plane" and "car" from label 0 and 1 to label 8 and 9 in train and test sets
        train_set['labels'] -= 2
        test_set['labels'] -= 2
        train_set['labels'][np.where(train_set['labels'] == -2)] = 8
        train_set['labels'][np.where(train_set['labels'] == -1)] = 9
        test_set['labels'][np.where(test_set['labels'] == -2)] = 8
        test_set['labels'][np.where(test_set['labels'] == -1)] = 9


    rng = np.random.RandomState(cfg.seed)

    #permute index of training set
    indices = rng.permutation(len(train_set['images']))
    train_set['images'] = train_set['images'][indices]
    train_set['labels'] = train_set['labels'][indices]

    #split training set into training and validation
    train_images = train_set['images'][n_valid:]
    train_labels = train_set['labels'][n_valid:]
    validation_images = train_set['images'][:n_valid]
    validation_labels = train_set['labels'][:n_valid]
    validation_set = {'images': validation_images, 'labels': validation_labels}
    train_set = {'images': train_images, 'labels': train_labels}

    #adopted from DSL3: get validaton and test dataset which have only ID samples
    validation_set = dataset.split_test(cfg, validation_set, n_class=n_class)
    test_set = dataset.split_test(cfg, test_set, n_class=n_class)
   #...............

    #split training set into labeled and unlabeled data
    l_train_set, u_train_set, indices_l_train_set, indices_u_train_set = dataset.split_l_u(cfg, train_set, n_labels, n_unlabels, n_class=n_class, ratio=ratio)

    print("Unlabeled data in distribuiton : {}, Unlabeled data out distribution : {}".format(
          np.sum(u_train_set['labels'] < n_class), np.sum(u_train_set['labels'] >= n_class)))


    #make an object of the Dataset class to give to  the Dataloder class which is from torch.util.Dataloader
    #https://github.com/perrying/pytorch-consistency-regularization/blob/6624c4e0bb1813b5952445ce34f9d4e52484ce38/ssl_lib/datasets/builder.py#L31 = dataset.SimpleDataset(l_train_set, transform)
    labeled_train_dataset = dataset.LabeledDataset(l_train_set)
    unlabeled_train_dataset = dataset.UnlabeledDataset(u_train_set)

    #train data is the concatenation of the labeled set and unlabeled set
    #we get the all train data to do preprocessing stasstics for zca
    train_data = np.concatenate([
        labeled_train_dataset.dataset["images"],
        unlabeled_train_dataset.dataset["images"]
    ], 0)

    # set augmentation on  train data(it should be implemented on dataset objects)
    flags = [True if b == "t" else False for b in cfg.wa.split(".")]
    #get zca parameter for zca normalisation
    if cfg.whiten:
        mean = train_data.mean((0, 1, 2)) / 255.
        scale = train_data.std((0, 1, 2)) / 255.
    elif cfg.zca:
        mean, scale = dataset.get_zca_normalization_param(numpy_batch_gcn(train_data))
    else:
        # from [0, 1] to [-1, 1]
        mean = [0.5, 0.5, 0.5]
        scale = [0.5, 0.5, 0.5]

    flags = [True if b == "t" else False for b in cfg.wa.split(".")]

    if cfg.labeled_aug == "RA":
        labeled_augmentation = gen_strong_augmentation(
            img_size, mean, scale, flags[0], flags[1], randauglist, cfg.zca)
    elif cfg.labeled_aug == "WA":
        labeled_augmentation = gen_weak_augmentation(cfg.img_size, mean, scale, *flags, cfg.zca)
    else:
        raise NotImplementedError

    labeled_train_dataset.transform = labeled_augmentation

    if cfg.unlabeled_aug == "RA":
        unlabeled_augmentation = gen_strong_augmentation(
            img_size, mean, scale, flags[0], flags[1], randauglist, cfg.zca)
    elif cfg.unlabeled_aug == "WA":
        unlabeled_augmentation = gen_weak_augmentation(cfg.img_size, mean, scale, *flags, cfg.zca)
    else:
        raise NotImplementedError

    if logger is not None:
        logger.info("labeled augmentation")
        logger.info(labeled_augmentation)
        logger.info("unlabeled augmentation")
        logger.info(unlabeled_augmentation)

    unlabeled_train_dataset.weak_augmentation = unlabeled_augmentation

    if cfg.zca:
        test_transform = transforms.Compose([GCN(), ZCA(mean, scale)])
    else:
        test_transform = transforms.Compose([transforms.Normalize(mean, scale, True)])

  #make test dataset and validataion dataset objects and aplly the  GCN and ZCA)
    test_dataset = dataset.LabeledDataset(test_set, test_transform)
    validation_dataset = dataset.LabeledDataset(validation_set, test_transform)

    print("labeled data : {}, unlabeled data : {},  training data : {}".format(
        len(labeled_train_dataset), len(unlabeled_train_dataset), len(labeled_train_dataset) + len(unlabeled_train_dataset)))
    print("validation data : {}, test data : {}".format(len(validation_dataset), len(test_dataset)))


    data_loaders = {
        'labeled': torch.utils.data.DataLoader(
            dataset=labeled_train_dataset, batch_size= l_batch_size, drop_last=True,
            sampler = InfiniteSampler(len(labeled_train_dataset), cfg.iteration * cfg.l_batch_size),
            num_workers=cfg.num_workers),
        'unlabeled': torch.utils.data.DataLoader(
            dataset=unlabeled_train_dataset, batch_size=ul_batch_size, drop_last=True,
            sampler = InfiniteSampler(len(unlabeled_train_dataset), cfg.iteration * cfg.ul_batch_size),
            num_workers=cfg.num_workers),

    #no need to shuffle the test datasets and validation
        'valid': torch.utils.data.DataLoader(
            dataset=validation_dataset, batch_size= 50, shuffle=False, drop_last=True, num_workers=cfg.num_workers),
        'test': torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=50, shuffle=False, drop_last=True, num_workers=cfg.num_workers)
    }
    return data_loaders

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)







