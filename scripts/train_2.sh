python train_2.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "resources/sk_feats/shift_ntu60_5_r" --wdir "results/shift_ntu60_5_r/" \
    --dis_step 4 --batch_size 32 --dataset ntu60
