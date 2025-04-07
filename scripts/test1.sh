alpha=0.8
alpha_p=0
body_part=4
n=4part
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== ntu60_5 with side info ====="
python train_7clf_s.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_$n" --wdir "results/shift_ntu60_5_r_$n/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p --body_part $body_part

alpha=0.8
alpha_p=0.3
body_part=4
n=4part
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== ntu60_5 with side info ====="
python train_7clf_s.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_$n" --wdir "results/shift_ntu60_5_r_$n/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p --body_part $body_part

alpha=0.8
alpha_p=0.5
body_part=4
n=4part
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== ntu60_5 with side info ====="
python train_7clf_s.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_$n" --wdir "results/shift_ntu60_5_r_$n/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p --body_part $body_part

alpha=0.8
alpha_p=0.8
body_part=4
n=4part
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== ntu60_5 with side info ====="
python train_7clf_s.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_$n" --wdir "results/shift_ntu60_5_r_$n/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p --body_part $body_part

alpha=0.8
alpha_p=1.0
body_part=4
n=4part
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== ntu60_5 with side info ====="
python train_7clf_s.py \
    --num_classes 60 --ss 5 --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size 96 --i_latent_size 8 --lr 4.9372938499672305e-05 --phase train --mode train --dataset_path "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_5_r_$n" --wdir "results/shift_ntu60_5_r_$n/" \
    --dis_step 4 --batch_size 32 --dataset ntu60 --alpha $alpha --alpha_p $alpha_p --body_part $body_part
