alpha=0.2
n=2
echo "alpha=$alpha"
echo "===== ntu60_5 with side info ====="
python train_f.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_dict" --wdir "results/shift_ntu60_5_r_dict_$n/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha 


echo "===== ntu60_12 with side info ====="
python train_f.py \
    --num_classes 60 --ss 12 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_12_r_dict" --wdir "results/shift_ntu60_12_r_dict_$n/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha 


echo "===== ntu120_10 with side info ====="
python train_f.py \
    --num_classes 120 --ss 10 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 304 --i_latent_size 12 --lr 2.133571484619993e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu120_10_r_dict" --wdir "results/shift_ntu120_10_r_dict_$n/" \
    --dis_step 16 --batch_size 24 --dataset ntu120 --alpha $alpha 
    

echo "===== ntu120_24 with side info ====="
python train_f.py \
    --num_classes 120 --ss 24 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 304 --i_latent_size 12 --lr 2.133571484619993e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu120_24_r_dict" --wdir "results/shift_ntu120_24_r_dict_$n/" \
    --dis_step 16 --batch_size 24 --dataset ntu120 --alpha $alpha


