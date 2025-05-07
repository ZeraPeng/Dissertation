# 5
echo "5"
python train_char_align_score.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 1.05e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_2part" --wdir "char_results/shift_ntu60_5_r/" \
    --dis_step 4 --batch_size 32 --dataset ntu60

# # 12
# echo "12"
# python train_char_align_score.py \
#     --num_classes 60 --ss 12 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 1.05e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_12_r_2part" --wdir "char_results/shift_ntu60_12_r/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60

# # 10
# echo "10"
# python train_char_align_score.py \
#     --num_classes 120 --ss 10 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 304 --i_latent_size 12 --lr 1.05e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu120_10_r_2part" --wdir "char_results/shift_ntu120_10_r/" \
#     --dis_step 16 --batch_size 24 --dataset ntu120

# # 24
# echo "24"
# python train_char_align_score.py \
#     --num_classes 120 --ss 24 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 304 --i_latent_size 12 --lr 2.05e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu120_24_r_2part" --wdir "char_results/shift_ntu120_24_r/" \
#     --dis_step 16 --batch_size 24 --dataset ntu120