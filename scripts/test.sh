#! /usr/bin/bash
st="r" results_dir="results"
visual_encoder="shift" language_encoder="clip-vit-b-32" tm="chat"
mode="train"
dataset=$1

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

ss="5"
dataset_local="ntu60"
tdir="resources/sk_feats/${visual_encoder}_${dataset_local}_${ss}_r"
edir="resources/sk_feats/${visual_encoder}_${dataset_local}_val_${ss}_r/"
wdir_1="results/${visual_encoder}_${dataset_local}_${ss}_r/"
wdir_2="results/${visual_encoder}_${dataset_local}_val_${ss}_r/"

echo "-----------------------------------"
echo "=========="
echo "Stage 1" # train
echo "..."
r1=$(
    python train_2.py \
        --num_classes $num_classes --ss $ss --st $st --ve $visual_encoder --le $language_encoder --tm $tm --num_cycles $nc --num_epoch_per_cycle $nepc \
        --latent_size $ls --i_latent_size $ils --lr $lr --phase train --mode $mode --dataset_path "$tdir" --wdir "$wdir_1" \
        --dis_step $dis_step --batch_size $batch_size --dataset $dataset_local
)
za=${r1:0-35:5} c=${r1:0-18:1}
echo "Best ZSL Acc: $za on cycle $c"
