# echo "===== ntu60_5 with side info ====="
# alpha=0.8
# alpha_p=0.5
# echo "alpha=$alpha, alpha_p=$alpha_p"
# echo "2 clf trained together maxpool."
# python train_2clf_t_maxpool.py \
#     --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_2clf_t_maxpool/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p 

# echo "===== ntu60_5 with side info ====="
# alpha=0.8
# alpha_p=0
# echo "alpha=$alpha, alpha_p=$alpha_p"
# echo "2 clf trained together average pool."
# python train_2clf_t_avepool.py \
#     --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_2clf_t_avepool/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p 

# echo "===== ntu60_5 with side info ====="
# alpha=0.8
# alpha_p=0.5
# echo "alpha=$alpha, alpha_p=$alpha_p"
# echo "2 clf trained separately maxpool."
# python train_2clf_s_maxpool.py \
#     --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_2clf_s_maxpool/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p 

# echo "===== ntu60_5 with side info ====="
# alpha=0.5
# alpha_p=0.5
# echo "alpha=$alpha, alpha_p=$alpha_p"
# echo "1 clf trained (MLP2) averagepool."
# python train_1clf.py \
#     --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_1clf/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p 

echo "===== ntu60_5 with side info ====="
alpha=0.75
alpha_p=1.0
echo "alpha=$alpha, alpha_p=$alpha_p"
echo "1 clf trained (2 streams)."
python train_1clf_2stream.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_1clf/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p 

# echo "===== ntu60_5 with side info ====="
# alpha=0.5
# alpha_p=0.5
# echo "1 clf trained (MLP) [2500, 7*96]."
# python train_1clf_stack.py \
#     --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_1clf/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p 

# echo "===== ntu60_5 origin ====="
# python train_newdata.py \
#     --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_origin/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60
