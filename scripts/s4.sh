echo "Stage 4"
python gating_eval.py \
    --num_classes 60 --ss "5" --st r --ve shift --le clip-vit-b-32 --tm chat --phase train --dataset_path  "/usr1/home/s124mdg53_04/Dissertation/resources/sk_feats/shift_ntu60_5_r" \
    --wdir "results/shift_ntu60_5_r_origin/" --thresh 0.53 --temp 8 --dataset ntu60