#python utils/prepare_ego4d_dataset.py \
#    --input_train_split /playpen-storage/mmiemon/ego4d/data/annotations/nlq_train.json \
#    --input_val_split /playpen-storage/mmiemon/ego4d/data/annotations/nlq_val.json \
#    --input_test_split /playpen-storage/mmiemon/ego4d/data/annotations/nlq_val.json \
#    --video_feature_read_path /playpen-storage/mmiemon/ego4d/data/v1/slowfast8x8_r101_k400/ \
#    --clip_feature_save_path data/features/nlq_official_v1/official \
#    --output_save_path data/dataset/nlq_official_v1

python main.py \
    --task nlq_official_v1 \
    --predictor bert \
    --mode train \
    --video_feature_dim 2304 \
    --max_pos_len 512 \
    --epochs 200 \
    --fv official \
    --num_workers 64 \
    --model_dir checkpoints/ \
    --eval_gt_json "data/nlq_val.json"