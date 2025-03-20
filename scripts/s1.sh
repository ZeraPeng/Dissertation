# python train_s2.py \
#     --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_1/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60 --load_classifier --load_vae

python train.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/Dissertation/resources/sk_feats/shift_ntu60_5_r" --wdir "results/shift_ntu60_5_r_origin/" \
    --dis_step 4 --batch_size 32 --dataset ntu60
