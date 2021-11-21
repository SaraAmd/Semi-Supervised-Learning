python train.py \
--lr 3e-2 \
-wd 5e-4 \
--dataset CIFAR10 \
--ul_batch_size 320 \
--l_batch_size 64 \
--weight_average \
--iteration 500000 \
--checkpoint 1024 \
--wa_apply_wd \
--alg pl \
--strong_aug \
--threshold 0.95 \
--coef 1 \
--out_dir $1 \
--n_labels $2