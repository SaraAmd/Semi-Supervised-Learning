# Semi-Supervised-Learning (SSL)
This repository contains the deep semi-supervised learning on CIFAR-10 dataset detecting out of distribution data points (OOD)
in unlabeled data in order to improve the performance of the SSL algorithms
The repository has the implementation of Two state of the art SSL algorithms:
Fixmatch (https://arxiv.org/abs/2001.07685) 
Unsupervised Data Augmentation UDA (https://arxiv.org/abs/1904.12848)

To run for the FixMatch type

sh ./scripts/fixmatch ./results/fixmatch.sh 2400 for Fixmatch and sh ./scripts/uda.sh ./results/uda 2400 for UDA method
(2400) refers to the number of  labeled data points abvailable for training

for UDA type

sh ./scripts/uda.sh ./results/uda 2400

