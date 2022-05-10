#python main.py \
#    --task nlq_official_clip_10s \
#    --predictor bert \
#    --mode test \
#    --video_feature_dim 1024 \
#    --max_pos_len 512 \
#    --epochs 200 \
#    --fv video_swin \
#    --num_workers 64 \
#    --model_dir checkpoints/nlq_official_clip_10s \
#    --eval_gt_json "/playpen-storage/mmiemon/ego4d/data/annotations/nlq_val_10s.json"


python main.py \
    --task nlq_official_clip_10s \
    --predictor bert \
    --mode test \
    --video_feature_dim 1024 \
    --dim 128 \
    --max_pos_len 512 \
    --fv video_swin \
    --nms_th 0.5 \
    --model_dir checkpoints/nlq_official_clip_10s \
    --eval_gt_json "/playpen-storage/mmiemon/ego4d/data/annotations/nlq_val_10s.json"