#ckpt_path=$1
#eval_split_name=$2
#eval_path=data/highlight_${eval_split_name}_release.jsonl
#PYTHONPATH=$PYTHONPATH:. python moment_detr/inference.py \
#--resume ${ckpt_path} \
#--eval_split_name ${eval_split_name} \
#--eval_path ${eval_path} \
#${@:3}


ckpt_path=/playpen-storage/mmiemon/ego4d/NLQ/moment_detr/results/ego4d-video_tef-queries_20_500/model_best.ckpt
eval_split_name=val
eval_path=data/ego4d_clip_10s/val_250.json
PYTHONPATH=$PYTHONPATH:. python moment_detr/inference_ego4d.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:3}

#python inference_debug.py --eval_split val --eval_path data/ego4d_nlq_moment_detr_val.json --resume /playpen-storage/mmiemon/ego4d/NLQ/moment_detr/results/ego4d-video_tef-clip_text_clip_video_250_l1_right/model_best.ckpt


#PYTHONPATH=$PYTHONPATH:. python inference_ego4d_debug.py \
#--resume /playpen-storage/mmiemon/ego4d/NLQ/moment_detr/results/all_data_mean_pool/clip_similarity/ego4d-video_tef-queries_20_vl_500/model_best.ckpt \
#--eval_split_name val \
#--eval_path data/ego4d_nlq_moment_detr_val_500.json
