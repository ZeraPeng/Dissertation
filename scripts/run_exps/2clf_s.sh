# echo "===== ntu60_5 with side info ====="
# alpha=0.8
# alpha_p=0
# echo "alpha=$alpha, alpha_p=$alpha_p"
# echo "2 clf trained separately."
# python train_2clf_s.py \
#     --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
#     --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_2clf_s/" \
#     --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p

echo "===== ntu60_5 with side info ====="
alpha=0.8
alpha_p=0.3
echo "alpha=$alpha, alpha_p=$alpha_p"
echo "2 clf trained separately."
python train_2clf_s.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_2clf_s/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p

echo "===== ntu60_5 with side info ====="
alpha=0.8
alpha_p=0.5
echo "alpha=$alpha, alpha_p=$alpha_p"
echo "2 clf trained separately."
python train_2clf_s.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_2clf_s/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p

echo "===== ntu60_5 with side info ====="
alpha=0.8
alpha_p=0.8
echo "alpha=$alpha, alpha_p=$alpha_p"
echo "2 clf trained separately."
python train_2clf_s.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_2clf_s/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p

echo "===== ntu60_5 with side info ====="
alpha=0.8
alpha_p=1.0
echo "alpha=$alpha, alpha_p=$alpha_p"
echo "2 clf trained separately."
python train_2clf_s.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_2clf_s/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p
