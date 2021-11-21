import argparse


def get_args():

    parser = argparse.ArgumentParser(description='SSL Implementation')

    #dataset config
    parser.add_argument("--root", "-r", default="./data", type=str, help="/path/to/dataset")
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--img_size', default=32)
    parser.add_argument("--whiten", action="store_true", help="use whitening as preprocessing")
    parser.add_argument("--zca", action="store_true", help="use zca whitening as preprocessing")
    parser.add_argument("--num_workers", default=8, type=int, help="number of thread for CPU parallel")
    #400 images per class to construct the labeled data set, i.e., 6*400= 2,400 labeled examples for CIFAR10.
    parser.add_argument('--n_labels', type=int, default=2400)
    parser.add_argument('--n_unlabels', type=int, default=20000)
    parser.add_argument('--n_valid', type=int, default=5000)
    parser.add_argument('--n_class',  help="number of in distribution (ID) classes", type=int, default=6)
    parser.add_argument('--tot_class', help="number of all the classes available in dataset", type=int, default=10)
    parser.add_argument('--warmup_iter', help="Number representing the warmup iterations using labelled datapoints", type=int, default=4000)

    #the percentage of the samples in unlabeled data that are OODs.
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--l_batch_size', type=int, default=50)
    parser.add_argument('--ul_batch_size', type=int, default=50)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--similarity', default='exp_cos', type=str, choices=['cosine', 'exp_cos'],
                    help='similarity metrics')
    #parser.add_argument('--iterations', type=int, default=20000)

    # augmentation config
    parser.add_argument("--labeled_aug", default="WA", choices=['WA', 'RA'], type=str,
                        help="type of augmentation for labeled data")
    parser.add_argument("--unlabeled_aug", default="WA", choices=['WA', 'RA'], type=str,
                        help="type of augmentation for unlabeled data")
    parser.add_argument("--wa", default="t.t.f", type=str,
                        help="transformations (flip, crop, noise) for weak augmentation. t and f indicate true and false.")
    parser.add_argument("--strong_aug", action="store_true",
                        help="use strong augmentation (RandAugment) for unlabeled data")

#Model
    parser.add_argument('--arch', default='wrn', type=str, help=' wrn')
    parser.add_argument("--alg", "-a", default="cr", type=str, help="ssl algorithm : [supervised, 'ict', 'cr', 'pl', 'vat']]")

# optimization config
    parser.add_argument("--optimizer", "-opt", default="sgd", choices=['sgd', 'adam'], type=str, help="optimizer")
    parser.add_argument("--lr", default=3e-2, type=float, help="learning rate")
    parser.add_argument("--weight_decay", "-wd", default=0.01, type=float, help="weight decay")

    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for sgd or beta_1 for adam")
    parser.add_argument("--iteration", default=500000, type=int, help="number of training iteration")
    parser.add_argument("--lr_decay", default="cos", choices=['cos', 'step', 'None'], type=str, help="way to decay learning rate")
    parser.add_argument("--lr_decay_rate", default=0.2, type=float, help="decay rate for step lr decay")
    parser.add_argument("--only_validation", action="store_true",
                    help="only training and validation for hyperparameter tuning")
    parser.add_argument("--tsa", action="store_true", help="use training signal annealing proposed by UDA")
    parser.add_argument("--tsa_schedule", default="linear", choices=['linear', 'exp', 'log'], type=str, help="tsa schedule")
    parser.add_argument("--warmup_iter_ssl", default=0, type=int,
                        help="wnumber of armup iteration for SSL loss coefficient")

#default is the same augmentation for all methods (moderate augmentation: random cropping, padding, whitening and horizontal flippin)
    #parser.add_argument('--augPolicy', default=1, type=int, help='augmentation policy: 0 for none, 1 for moderate, 2 for heavy (random-augment)')
    parser.add_argument('--use_zca', dest='use_zca', action='store_true',
                    help='use zca whitening')

# SSL common config
    parser.add_argument("--coef", default=1, type=float, help="coefficient for consistency loss")
    parser.add_argument("--coef_schaduler", default=True, type=bool, help="schaduler for the coefficient for consistency loss")
    #parser.add_argument("--ema", action="store_true", help="Stochastic Moving average")


    parser.add_argument( "--consistency", default="ce", choices=['ce', 'ms'], type=str, help="consistency type, cross-entropy, mean squre")
    parser.add_argument("--entropy_minimization", "-em", default=0, type=float,
                        help="coefficient of entropy minimization")

    parser.add_argument("--ema_teacher", action="store_true", help="use mean teacher")
    parser.add_argument("--ema_teacher_warmup", action="store_true", help="warmup for mean teacher")
    parser.add_argument("--ema_teacher_factor", default=0.999, type=float,
                        help="exponential mean avarage factor for mean teacher")
    parser.add_argument("--ema_apply_wd", action="store_true", help="apply weight decay to ema model")

    parser.add_argument("--threshold", default=None, type=float, help="pseudo label threshold")
    parser.add_argument("--sharpen", default=None, type=float, help="tempereture parameter for sharpening")
    parser.add_argument("--temp_softmax", default=None, type=float, help="tempereture for softmax")

## SSL alg parameter
### ICT config
    parser.add_argument("--alpha", default=0.1, type=float, help="parameter for beta distribution in ICT")
### VAT config
    parser.add_argument("--eps", default=6, type=float, help="norm of virtual adversarial noise")
    parser.add_argument("--xi", default=1e-6, type=float, help="perturbation for finite difference method")
    parser.add_argument("--vat_iter", default=1, type=int, help="number of iteration for power iteration")
# evaluation config
    parser.add_argument("--weight_average", action="store_true", help="evaluation with weight-averaged model")
    parser.add_argument("--wa_ema_factor", default=0.999, type=float, help="exponential mean avarage factor for weight-averaged model")
    parser.add_argument("--wa_apply_wd", action="store_true", help="apply weight decay to weight-averaged model")
    parser.add_argument("--checkpoint", default=10000, type=int, help="checkpoint every N samples")
    # training from checkpoint
    parser.add_argument("--checkpoint_model", default=None, type=str, help="path to checkpoint model")
    parser.add_argument("--checkpoint_optimizer", default=None, type=str, help="path to checkpoint optimizer")
    parser.add_argument("--start_iter", default=None, type=int, help="start iteration")
    # misc
    parser.add_argument("--out_dir", default="log", type=str, help="output directory")
    parser.add_argument("--seed", default=96, type=int, help="random seed")
    parser.add_argument("--disp", default=256, type=int, help="display loss every N")


    return  parser.parse_args()