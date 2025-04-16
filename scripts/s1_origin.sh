# python train_f.py \
#     --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 20 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60 

# ntu60_5 origin
python train.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/Dissertation/resources/sk_feats/shift_ntu60_5_r" --wdir "results/shift_ntu60_5_r_origin/" \
    --dis_step 4 --batch_size 32 --dataset ntu60


# ntu60_12 with side info
# python train_f.py \
#     --num_classes 60 --ss 12 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 20 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_12_r_dict" --wdir "results/shift_ntu60_12_r_dict_1/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60 --load_classifier

# ntu60_12 origin
# echo "===== ntu60_12 origin ====="
# python train_newdata.py \
#     --num_classes 60 --ss 12 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_12_r_6part" --wdir "results/shift_ntu60_12_r_origin/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60

# echo "===== ntu120_10 with side info ====="
# python train_f.py \
#     --num_classes 120 --ss 10 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 2.133571484619993e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu120_10_r_dict" --wdir "results/shift_ntu120_10_r_dict/" \
#     --dis_step 16 --batch_size 24 --dataset ntu120 --load_classifier

echo "===== ntu120_10 origin ====="
python train_newdata.py \
    --num_classes 120 --ss 10 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 304 --i_latent_size 12 --lr 2.133571484619993e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu120_10_r_6part" --wdir "results/shift_ntu120_10_r_origin/" \
    --dis_step 16 --batch_size 24 --dataset ntu120

# echo "===== ntu120_24 with side info ====="
# python train_f.py \
#     --num_classes 120 --ss 24 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 2.133571484619993e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu120_24_r_dict" --wdir "results/shift_ntu120_24_r_dict/" \
#     --dis_step 16 --batch_size 24 --dataset ntu120 

echo "===== ntu120_24 origin ====="
python train_newdata.py \
    --num_classes 120 --ss 24 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 304 --i_latent_size 12 --lr 2.133571484619993e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu120_24_r_6part" --wdir "results/shift_ntu120_24_r_origin/" \
    --dis_step 16 --batch_size 24 --dataset ntu120