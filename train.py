import  torch
import  json
from scipy import spatial
import load
import time
import torch.nn.functional as F
import numpy
import random
import logging
import torch.optim as optim

from ssl_lib.consistency.builder import gen_consistency
from ssl_lib.algs.builder import gen_ssl_alg
from ssl_lib.models.builder import gen_model
from ssl_lib.misc.meter import Meter
from ssl_lib.param_scheduler import scheduler
from ssl_lib.models import utils as model_utils
from ssl_lib.algs import utils as alg_utils

def average_features(feature_vectors_mapping, n_class):
    featureVector = []
    for i in feature_vectors_mapping.keys():
        featureVector.append(feature_vectors_mapping[i][0] / feature_vectors_mapping[i][1])
        #print("the number of features in class",i, " is: ", feature_vectors_mapping[i][1])
    return(featureVector)


def evaluation(raw_model, eval_model, loader, device):
    raw_model.eval()
    eval_model.eval()
    sum_raw_acc = sum_acc = sum_loss = 0
    with torch.no_grad():
        for (data, labels) in loader:
            data, labels = data.to(device), labels.to(device)

            #forward pass
            preds = eval_model(data)
            raw_preds = raw_model(data)
            #softmax comes with cross entropy loss for numerical stability in the pytorch
            loss = F.cross_entropy(preds, labels)
            sum_loss += loss.item()
            #get max predicton over axis one which means we take the max over the columns, returns maximum value in each row:
            #the outout of max is the index of the maximum prediction (second element)
            # and the prediction value the (first element), output : ( max value of presiction, index)
            # we need the index which coresponds to the class
            acc = (preds.max(1)[1] == labels).float().mean()
            raw_acc = (raw_preds.max(1)[1] == labels).float().mean()
            #updatae count
            sum_acc += acc.item()
            sum_raw_acc += raw_acc.item()
    mean_raw_acc = sum_raw_acc / len(loader)
    mean_teacher_acc = sum_acc / len(loader)
    mean_loss_of_mean_techer = sum_loss / len(loader)
    raw_model.train()
    eval_model.train()
    return mean_raw_acc, mean_teacher_acc, mean_loss_of_mean_techer

'''
:param:labeled: is the labele of labeled data not psudolabels
avarage_model: will be the same as eval model if we do not use exponential moving average for evaluaton
'''


def param_update(
    cfg,
    cur_iteration,
    model,
    teacher_model,
    optimizer,
    ssl_alg,
    consistency,
    labeled_data,
    ul_weak_data,
    ul_strong_data,
    labels,
    average_model,
    warmup_number,
active_meantecher
):

    #measure the time of one iteration
    start_time = time.time()

    #concatenate all labeled data and unlabeled data
    if(cur_iteration < warmup_number):
        all_data = labeled_data
    else:
        all_data = torch.cat([labeled_data, ul_weak_data, ul_strong_data], 0)

    forward_func = model.forward
    #get the model prediction
    stu_logits = model(all_data)
    #features = model.features()
    # get prediction for labeled data
    labeled_preds = stu_logits[:labeled_data.shape[0]]

    #get prediction for unlabled data
    #stu_unlabeled_weak_logits = stu_logits[labels.shape[0]:]
    stu_unlabeled_weak_logits, stu_unlabeled_strong_logits = torch.chunk(stu_logits[labels.shape[0]:], 2, dim=0)

    # compute the supervised loss
    #supervised loss for UDA method
    if cfg.tsa and cur_iteration >= warmup_number:
        none_reduced_loss = F.cross_entropy(labeled_preds, labels, reduction="none")
        L_supervised = alg_utils.anneal_loss(
            labeled_preds, labels, none_reduced_loss, cur_iteration+1,
            cfg.iteration, labeled_preds.shape[1], cfg.tsa_schedule)
    else:
        L_supervised = F.cross_entropy(labeled_preds, labels)


    #if warm up is done, compute the unsupervsed loss
    if(cfg.coef > 0 and cur_iteration >= warmup_number ):


        # get target values from teacher model
        if teacher_model is not None:
            t_forward_func = teacher_model.forward
            tea_logits = t_forward_func(all_data)
            tea_unlabeled_weak_logits, _ = torch.chunk(tea_logits[labels.shape[0]:], 2, dim=0)
        else:
            t_forward_func = forward_func
            tea_unlabeled_weak_logits = stu_unlabeled_weak_logits

        model.update_batch_stats(False)

        # calc consistency loss
# ssl_alg  return ConsistencyRegularization and ConsistencyRegularization reurns stu_preds, adjusted targets, mask for psuldo labeling
        y, targets, mask = ssl_alg(
            stu_preds = stu_unlabeled_strong_logits,
            tea_logits = tea_unlabeled_weak_logits.detach(),
            data = ul_strong_data,
            stu_forward = forward_func,
            tea_forward = t_forward_func
            )
        #model.update_batch_stats(True)
        #returns the loss from for example CrossEntropy class which returns
        #consistency is consistency type
        #L_consistency = consistency(y, targets, mask, weak_prediction=stu_unlabeled_weak_logits.softmax(1))
        model.update_batch_stats(True)
        L_consistency = consistency(y, targets, mask, weak_prediction=tea_unlabeled_weak_logits.softmax(1))
        # schaduler for coef of unsupervised loss
        if cfg.coef_schaduler:
            coef = scheduler.linear_warmup(cfg.coef, cfg.warmup_iter, cur_iteration + 1)
        else:
            coef = cfg.coef
   #supervised learning

    else:
        L_consistency = torch.zeros_like(L_supervised)
        coef = 0
        mask = None

    # calc total loss
    loss = L_supervised + coef * L_consistency


    if cfg.entropy_minimization > 0:
        loss -= cfg.entropy_minimization * \
                    (stu_unlabeled_weak_logits.softmax(1) * F.log_softmax(stu_unlabeled_weak_logits, 1)).sum(1).mean()

    # update parameters
    #get access to current learning rate
    cur_lr = optimizer.param_groups[0]["lr"]
    #zero the parameter gradients
    optimizer.zero_grad()
    #take the deravitives(gradients) and backward
    loss.backward()
    #if we have weight regularization in the loss
    #weight decay factor 0.2 after 400,000 iterations
    if cfg.weight_decay > 0 and cur_iteration >= warmup_number:
        decay_coeff = cfg.weight_decay * cur_lr
        model_utils.apply_weight_decay(model.modules(), decay_coeff)
    # update the parameters
    optimizer.step()
    # update the average model and teacher model during the warmup
    # if cur_iteration == cfg.warmup_iter:
    #     for ave_p, raw_p in zip(average_model.parameters(), model.parameters()):
    #         ave_p.data.copy_(raw_p.data)
    #     if cfg.ema_teacher:
    #         for tea_p, raw_p in zip(teacher_model.parameters(), model.parameters()):
    #             tea_p.data.copy_(raw_p.data)

    # update teacher model's parameters by exponential moving average
    if cfg.ema_teacher and cur_iteration >= cfg.warmup_iter:
        model_utils.ema_update(
            teacher_model, model, cfg.wa_ema_factor, cfg.weight_decay * cur_lr if cfg.wa_apply_wd else None
        )

    # update evaluation model's parameters by exponential moving average
    if cfg.weight_average and cur_iteration >= cfg.warmup_iter:
        model_utils.ema_update(
            average_model, model, cfg.wa_ema_factor, cfg.weight_decay * cur_lr if cfg.wa_apply_wd else None
        )
    # calculate accuracy for labeled data
    acc = (labeled_preds.max(1)[1] == labels).float().mean()


    return {
            "labeled_acc": acc,
            "loss": loss.item(),
            "sup loss": L_supervised.item(),
            "ssl loss": L_consistency.item(),
            "mask": mask.float().mean().item() if mask is not None else 1,
            "coef": coef,
            "sec/iter": (time.time() - start_time)
    }


def main(cfg, logger):

    # set seed
    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    #torch.cuda.manual_seed(cfg.seed)
    #torch.cuda.manual_seed_all(cfg.seed)
    #backends.cudnn.deterministic = True
    #all the model parameters and the input data should be on the same gpu or RAM
    # select device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benckmark = True
        print("running on GPU!")
    else:
        logger.info("CUDA is NOT available")
        device = "cpu"
        print("running on CPU")

    # build data loader
    logger.info("load dataset")
    data_loaders = load.get_dataloaders(root= cfg.root, data=cfg.dataset, n_labels=cfg.n_labels, n_unlabels=cfg.n_unlabels, n_valid=cfg.n_valid,
                                        l_batch_size=cfg.l_batch_size, ul_batch_size=cfg.ul_batch_size,
                                        test_batch_size=cfg.test_batch_size, iterations=cfg.iteration,
                                        n_class=cfg.n_class, ratio=cfg.ratio, unlabeled_aug=cfg.unlabeled_aug, logger=logger, cfg=cfg)
    label_loader = data_loaders['labeled']
    unlabel_loader = data_loaders['unlabeled']
    test_loader = data_loaders['test']
    val_loader = data_loaders['valid']
    num_classes = cfg.n_class
    img_size = cfg.img_size
    print("data is loaded!")

    active_meantecher =False
    # set consistency type: consistency type (cross entropy, mean squre)
    consistency = gen_consistency(cfg.consistency, cfg)
    # set ssl algorithm
    ssl_alg = gen_ssl_alg(cfg.alg, cfg)

    # build student model
    model = gen_model(cfg.arch, num_classes, img_size).to(device)
    # for ema build the teacher model
    if cfg.ema_teacher:
        teacher_model = gen_model(cfg.arch, num_classes, img_size).to(device)
        teacher_model.load_state_dict(model.state_dict())
    else:
        teacher_model = None
    # for evaluation
    if cfg.weight_average:
        average_model = gen_model(cfg.arch, num_classes, img_size).to(device)
        average_model.load_state_dict(model.state_dict())
    else:
        average_model = None

# sets the model in training mode (it does not train the model)
    model.train()

    logger.info(model)

    # build optimizer
    if cfg.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), cfg.lr, cfg.momentum, weight_decay=0, nesterov=True
        )
    elif cfg.optimizer == "adam":
        optimizer = optim.AdamW(
            model.parameters(), cfg.lr, (cfg.momentum, 0.999), weight_decay=0
        )
    else:
        raise NotImplementedError
    # set lr scheduler
    if cfg.lr_decay == "cos":
        lr_scheduler = scheduler.CosineAnnealingLR(optimizer, cfg.iteration)
    elif cfg.lr_decay == "step":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [400000, ], cfg.lr_decay_rate)
    else:
        raise NotImplementedError

    # init meter
    metric_meter = Meter()
    test_acc_list = []
    raw_acc_list = []
    maximum_val_acc = 0
    logger.info("training")

    for i,(l_data, ul_data) in enumerate(zip(label_loader, unlabel_loader)):


        l_aug, labels = l_data
        ul_w_aug, ul_s_aug, u_labels = ul_data

        l_batchSize = len(labels)
        ul_batchSize = len(u_labels)
        #count the number of corect or incorect OODs detected by the model
        corect=0
        incorect=0

        #after warm up
        if (i>cfg.warmup_iter):

            # concatenate all labeled data and unlabeled data
            all_data = torch.cat([l_aug, ul_w_aug, ul_s_aug], 0).to(device)
            #give  the mdoel the data within a batch to extract features
            stu_logits = model(all_data)
            # extract features
            features = model.feature_extractor()
            #print("features ", features.size())
            #features = F.normalize(features)
            #print(" features ", features[1])
            #This is storing  the sum of features #for a particular class, along with the number of items added
            #So that while averaging we can have the number to divide it with
            feature_vectors_mapping = {}

            for j in range(l_batchSize):
                labelIdx = labels[j].item()
                if(labelIdx in feature_vectors_mapping.keys()):
                    feature_vectors_mapping[labelIdx][0] += features[j]
                    feature_vectors_mapping[labelIdx][1] += 1
                else:
                    feature_vectors_mapping[labelIdx] = [features[j],1]

            #Use this function call to get the average features across classes at any instance -> (feature_vectors_mapping)
            anchor_features = average_features(feature_vectors_mapping, cfg.n_class)

            cos_sim_ul = []

            for j in range(l_batchSize,l_batchSize+ul_batchSize):
                #cos_sim = -100000000
                sim = []
                for anchor in anchor_features:
                    #print(anchor)

                    # experimont with similarity mertrics
                    if cfg.similarity == 'exp_cos':
                        #print("the type of anchor is: ",torch.flatten(anchor).dtype)
                        #print("the type of features[j] is: ", torch.flatten(features[j]).dtype)
                        temp = F.cosine_similarity(torch.flatten(anchor),torch.flatten(features[j]), dim = 0)
                        #print('temp ', (temp))
                        temperature=1
                        similarity = torch.exp(temp) / temperature
                        #print('similarity ', (similarity))
                        sim.append(similarity)
                    else:
                        similarity = F.cosine_similarity(torch.flatten(anchor), torch.flatten(features[j]), dim=0)
                        sim.append(similarity)
                    #euclidena_dist = sum(((torch.flatten(anchor)-torch.flatten(features[j])) ** 2))
                    #find the maximum similarity between the unlabelded data poin to the anchors
                    # if(similarity > cos_sim):
                    #      cos_sim = similarity
                #print(" max sim ", max(sim))
                cos_sim_ul.append(max(sim))
            #print('len of cos_sim_ul ', len(cos_sim_ul))
            #print('len of cos_sim_ul ', (len(cos_sim_ul)))

            #get the standard deviaiton of similarities of Unlabele data with anchors within the batch
            sd = torch.std(torch.tensor(cos_sim_ul, dtype=float),unbiased =True).item()
            #print("sd of similarity is: ", sd)
            #get the mean of similarities of Unlabele data with anchors within the batch
            mean = torch.mean(torch.tensor(cos_sim_ul), dtype=float)
            #print("mean of similarity is: ", mean)
            #find the indices of OODs
            ood_indices = []
            ood_labels = []
            for j in range(len(cos_sim_ul)):
                if(cos_sim_ul[j]<mean-2*sd):
                    ood_indices.append(j)
                    ood_labels.append(u_labels[j])

            #print the perfrmance of the model for detecting the OODs
            real_OOD=0
            if i % 2000==0:
                for label in u_labels:
                    if label.item() == 6 or label .item() == 7 or label .item() == 8 or label .item() == 9:
                        real_OOD+=1
                print("the number of OOds in the batch  is: ", real_OOD)
                if len(ood_indices) !=0:
                    for label in ood_labels:
                        if label.item() == 6 or label.item() == 7 or label.item() == 8 or label.item() == 9:
                            corect+=1
                        else:
                            incorect+=1
                print("the algorithm detected ", corect, " of OOds")
                print("the algorithm detected ", incorect, " of IDs as OODs")

            # To remove the OOD dataset from the unlabeled
            for j in range(len(ood_indices)-1,-1,-1):

                ul_w_aug = torch.cat([ul_w_aug[:ood_indices[j]],ul_w_aug[ood_indices[j]+1:]],0)
                ul_s_aug = torch.cat([ul_s_aug[:ood_indices[j]], ul_s_aug[ood_indices[j] + 1:]], 0)
                ul_w_aug_labels = torch.cat([u_labels[:ood_indices[j]],u_labels[ood_indices[j]+1:]],0)


            #after filtering the OODs update the model (do the one iteration of training
            params = param_update(
                cfg, i, model, teacher_model, optimizer, ssl_alg,
                consistency, l_aug.to(device), ul_w_aug.to(device),
                ul_s_aug.to(device), labels.to(device),
                average_model, cfg.warmup_iter, active_meantecher
            )
        #do the warm up on labeled data
        else:
            params = param_update(
                cfg, i, model, teacher_model, optimizer, ssl_alg,
                consistency, l_aug.to(device), ul_w_aug.to(device),
                ul_s_aug.to(device), labels.to(device),
                average_model, cfg.warmup_iter,active_meantecher
            )
        #regular SSl
        # params = param_update(
        #     cfg, i, model, teacher_model, optimizer, ssl_alg,
        #     consistency, l_aug.to(device), ul_w_aug.to(device),
        #     ul_s_aug.to(device), labels.to(device),
        #     average_model, cfg.warmup_iter, active_meantecher
        # )



        # moving average for reporting losses and accuracy
        metric_meter.add(params, ignores=["coef"])

         # display losses every cfg.disp iterations
        if ((i+1) % cfg.disp) == 0:
            state = metric_meter.state(
                    header = f'[{i+1}/{cfg.iteration}]',
                    footer = f'ssl coef {params["coef"]:.4g} | lr {optimizer.param_groups[0]["lr"]:.4g}'
                )
            logger.info(state)

        lr_scheduler.step()
        # validation
        if ((i + 1) % cfg.checkpoint) == 0 or (i + 1) == cfg.iteration:
            with torch.no_grad():
                if cfg.weight_average:
                    eval_model = average_model
                else:
                    eval_model = model
                logger.info("validation")
                mean_raw_acc, mean_val_acc, mean_val_loss = evaluation(model, eval_model, val_loader, device)
                logger.info("validation loss %f | validation acc. %f | raw acc. %f", mean_val_loss, mean_val_acc,
                            mean_raw_acc)
                # if cfg.ema_teacher:
                #     mean_raw_acc, mean_val_acc, mean_val_loss = evaluation(model, teacher_model, val_loader, device)
                #     logger.info(" ema_model validation loss %f | ema_model validation acc. %f | raw_model validation acc. %f", mean_val_loss, mean_val_acc,
                #             mean_raw_acc)

                # test
                # if not cfg.only_validation and mean_val_acc > maximum_val_acc:
                #     torch.save(eval_model.state_dict(), os.path.join(cfg.out_dir, "best_model.pth"))
                #     maximum_val_acc = mean_val_acc
                #     logger.info("test")
                #     mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader, device)
                #     logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc,
                #                 mean_raw_acc)
                #     logger.info("test accuracy %f", mean_test_acc)
                logger.info("test")
                mean_raw_acc, mean_test_acc, mean_test_loss = evaluation(model, eval_model, test_loader, device)
                logger.info("test loss %f | test acc. %f | raw acc. %f", mean_test_loss, mean_test_acc, mean_raw_acc)
                test_acc_list.append(mean_test_acc)
                raw_acc_list.append(mean_raw_acc)

            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_checkpoint.pth"))
            torch.save(optimizer.state_dict(), os.path.join(cfg.out_dir, "optimizer_checkpoint.pth"))

    numpy.save(os.path.join(cfg.out_dir, "results"), test_acc_list)
    numpy.save(os.path.join(cfg.out_dir, "raw_results"), raw_acc_list)
    accuracies = {}
    for i in [1, 10, 20, 50]:
        logger.info("mean test acc. over last %d checkpoints: %f", i, numpy.median(test_acc_list[-i:]))
        logger.info("mean test acc. for raw model over last %d checkpoints: %f", i, numpy.median(raw_acc_list[-i:]))
        accuracies[f"last{i}"] = numpy.median(test_acc_list[-i:])

    with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
        json.dump(accuracies, f, sort_keys=True)


if __name__ == "__main__":
    import os, sys
    from parser import get_args
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # setup logger
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    s_handler = logging.StreamHandler(stream=sys.stdout)
    s_handler.setFormatter(plain_formatter)
    s_handler.setLevel(logging.DEBUG)
    logger.addHandler(s_handler)
    f_handler = logging.FileHandler(os.path.join(args.out_dir, "console.log"))
    f_handler.setFormatter(plain_formatter)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)
    logger.propagate = False

    logger.info(args)

    main(args, logger)





