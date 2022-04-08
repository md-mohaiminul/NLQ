python main.py \
    --task nlq_official_v1 \
    --predictor bert \
    --mode train \
    --video_feature_dim 768 \
    --max_pos_len 512 \
    --epochs 200 \
    --fv clip \
    --num_workers 64 \
    --model_dir checkpoints/ \
    --eval_gt_json "data/nlq_val.json"

#python main.py --task nlq_official_v1 --predictor bert --mode train --video_feature_dim 768 --max_pos_len 1024 --epochs 200 --fv clip --num_workers 64 --model_dir checkpoints/ --eval_gt_json "data/nlq_val.json"