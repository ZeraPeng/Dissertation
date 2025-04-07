dataset=ntu60
ss=12
alpha=0.8
body_part=2
dataset_path="/usr1/home/s124mdg53_04/STAR/packed_features/shift_${data_split}_${ss}_r_${body_part}part"
dataset_path="/usr1/home/s124mdg53_04/STAR/packed_features/shift_${data_split}_${ss}_r_${body_part}part"
w_dir="results/shift_${data_split}_${ss}_r_${body_part}par/"

if [ "$dataset" = "ntu60" ]; then
    ls=96 ils=8 lr=4.9372938499672305e-05 batch_size=32 dis_step=4
    th=0 t=0
    num_classes=60 nc=10 nepc=1700
    ss=5
    available_splits=("split2" "split3" "split4")
elif [ "$dataset" = "ntu120" ]; then
    ls=304 ils=12 lr=2.133571484619993e-05 batch_size=24 dis_step=16
    th=0 t=0
    num_classes=120 nc=10 nepc=1700
    ss=10
    available_splits=("split2" "split3" "split4")
else
    echo "Dataset not supported"
    exit 1
fi

echo "===data split: $data_split"
alpha_p=0
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== $data_split with side info ====="
python train_7clf_t.py \
    --num_classes $number_classes --ss $ss --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size $ls --i_latent_size $ils --lr $lr --phase train --mode train --dataset_path "$dataset_path" --wdir $w_dir \
    --dis_step $dis_step --batch_size $batch_size --dataset $dataset --alpha $alpha --alpha_p $alpha_p --body_part $body_part

alpha_p=0.3
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== $data_split with side info ====="
python train_7clf_t.py \
    --num_classes $number_classes --ss $ss --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size $ls --i_latent_size $ils --lr $lr --phase train --mode train --dataset_path "$dataset_path" --wdir $w_dir \
    --dis_step $dis_step --batch_size $batch_size --dataset $dataset --alpha $alpha --alpha_p $alpha_p --body_part $body_part

alpha_p=0.5
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== $data_split with side info ====="
python train_7clf_t.py \
    --num_classes $number_classes --ss $ss --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size $ls --i_latent_size $ils --lr $lr --phase train --mode train --dataset_path "$dataset_path" --wdir $w_dir \
    --dis_step $dis_step --batch_size $batch_size --dataset $dataset --alpha $alpha --alpha_p $alpha_p --body_part $body_part

alpha_p=0.8
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== $data_split with side info ====="
python train_7clf_t.py \
    --num_classes $number_classes --ss $ss --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size $ls --i_latent_size $ils --lr $lr --phase train --mode train --dataset_path "$dataset_path" --wdir $w_dir \
    --dis_step $dis_step --batch_size $batch_size --dataset $dataset --alpha $alpha --alpha_p $alpha_p --body_part $body_part

alpha_p=1.0
echo "alpha=$alpha, alpha_p=$alpha_p, body_part=$body_part"
echo "===== $data_split with side info ====="
python train_7clf_t.py \
    --num_classes $number_classes --ss $ss --st r --ve shift --le clip-vit-b-32 --tm chat --num_cycles 10 --num_epoch_per_cycle 1700 \
    --latent_size $ls --i_latent_size $ils --lr $lr --phase train --mode train --dataset_path "$dataset_path" --wdir $w_dir \
    --dis_step $dis_step --batch_size $batch_size --dataset $dataset --alpha $alpha --alpha_p $alpha_p --body_part $body_part
