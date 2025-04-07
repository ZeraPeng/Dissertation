#!/bin/bash

n=2
# 定义 alpha 的不同取值
alphas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# 遍历 alpha 并运行 Python 脚本
for alpha in "${alphas[@]}"; do
    echo "alpha=$alpha"
    echo "===== ntu60_12 with side info ====="
    python train_f.py \
    --num_classes 60 --ss 12 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_12_r_dict" --wdir "results/shift_ntu60_12_r_dict_$n/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --load_vae --load_classifier
done
