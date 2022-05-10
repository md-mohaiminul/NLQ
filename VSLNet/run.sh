python utils/prepare_ego4d_dataset.py \
    --input_train_split /playpen-storage/mmiemon/ego4d/data/annotations/nlq_train_10s.json \
    --input_val_split /playpen-storage/mmiemon/ego4d/data/annotations/nlq_val_10s.json \
    --input_test_split /playpen-storage/mmiemon/ego4d/data/annotations/nlq_val_10s.json \
    --output_save_path data/dataset/nlq_official_clip_10s
#    --video_feature_read_path /playpen-storage/mmiemon/ego4d/data/v1/slowfast8x8_r101_k400/ \
#    --clip_feature_save_path data/features/nlq_official_v1/official \

#python main.py \
#    --task nlq_official_v1 \
#    --predictor clip \
#    --mode train \
#    --video_feature_dim 768 \
#    --max_pos_len 128 \
#    --epochs 200 \
#    --fv clip \
#    --num_workers 64 \
#    --model_dir checkpoints/ \
#    --eval_gt_json "data/nlq_val.json"

#python main.py \
#    --task nlq_official_v1 \
#    --predictor bert \
#    --mode train \
#    --video_feature_dim 2304 \
#    --max_pos_len 128 \
#    --epochs 200 \
#    --fv official \
#    --num_workers 64 \
#    --model_dir checkpoints/ \
#    --eval_gt_json "data/nlq_val.json"

#best result: checkpoints/vslnet_nlq_official_v1_video_swin_512_bert
python main.py \
    --task nlq_official_v1 \
    --predictor bert \
    --mode test \
    --video_feature_dim 1024 \
    --dim 128 \
    --max_pos_len 512 \
    --fv video_swin \
    --model_dir checkpoints/ \
    --nms_th 0.5 \
    --eval_gt_json "data/nlq_val.json"

#python main.py \
#    --task nlq_official_v1 \
#    --predictor clip \
#    --mode test \
#    --video_feature_dim 768 \
#    --dim 128 \
#    --max_pos_len 512 \
#    --fv clip \
#    --model_dir checkpoints/ \
#    --eval_gt_json "data/nlq_val.json"
