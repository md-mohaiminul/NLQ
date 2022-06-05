#python utils/prepare_ego4d_dataset.py \
#    --input_train_split /playpen-storage/mmiemon/ego4d/data/annotations/nlq_train.json \
#    --input_val_split /playpen-storage/mmiemon/ego4d/data/annotations/nlq_val.json \
#    --input_test_split /playpen-storage/mmiemon/ego4d/Ego4d/annotations/v1/annotations/nlq_test_unannotated.json \
#    --output_save_path data/dataset/nlq_official_v1 \
#    --video_feature_read_path /playpen-storage/mmiemon/ego4d/data/v1/clip \
#    --clip_feature_save_path data/features/nlq_official_v1/clip \

#checkpoints/clip_128_512_nce/model/vslnet_35200.t7
#checkpoints/clip_128_512_nce/model/vslnet_35200_test_result.json

# To train the model.
CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp clip_video_swin_256_512_nce \
    --task nlq_official_v1 \
    --predictor clip \
    --bert_type base \
    --mode test \
    --video_feature_dim 1792 \
    --dim 256 \
    --max_pos_len 512 \
    --batch_size 32 \
    --epochs 200 \
    --fv video_swin \
    --num_workers 64 \
    --model_dir checkpoints/ \
    --eval_gt_json "/playpen-storage/mmiemon/ego4d/Ego4d/annotations/v1/annotations/nlq_val.json"

#checkpoints/vslnet_nlq_official_v1_video_swin_512_bert/model/vslnet_43824_test_result.json

#PRED_FILE="checkpoints/clip_128_512_nce/model"
#python utils/evaluate_ego4d_nlq.py \
#    --ground_truth_json /playpen-storage/mmiemon/ego4d/Ego4d/annotations/v1/annotations/nlq_val.json \
#    --model_prediction_json checkpoints/clip_128_512_nce/model/vslnet_35200_test_result.json \
#    --thresholds 0.3 0.5 \
#    --topK 1 3 5


# To predict on test set.
# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --task nlq_official_v1 \
#     --predictor bert \
#     --mode test \
#     --video_feature_dim 2304 \
#     --max_pos_len 128 \
#     --fv official \
#     --model_dir checkpoints/


# To evaluate predictions using official evaluation script.
# PRED_FILE="checkpoints/vslnet_nlq_official_v1_official_128_bert/model"
# python utils/evaluate_ego4d_nlq.py \
#     --ground_truth_json data/nlq_test.json \
#     --model_prediction_json ${PRED_FILE}/vslnet_43824_test_result.json \
#     --thresholds 0.3 0.5 \
#     --topK 1 3 5
