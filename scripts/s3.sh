echo "Stage 3"
# python gating_train.py \
#     --num_classes 60 --ss "5" --st r --ve shift --le clip-vit-b-32 --tm chat --phase val --dataset_path  "/usr1/home/s124mdg53_04/Dissertation/resources/sk_feats/shift_ntu60_val_5_r" \
#     --wdir "results/shift_ntu60_val_5_r_origin/" --th 0 --t 0 --dataset ntu60

python gating_train.py \
    --num_classes 60 --ss "5" --st r --ve shift --le clip-vit-b-32 --tm chat --phase val --dataset_path  "/usr1/home/s124mdg53_04/STAR/packed_features/shift_ntu60_val_5_r_dict" \
    --wdir "results/shift_ntu60_val_5_r_dict/" --th 0 --t 0 --dataset ntu60