dset_name=ego4d
ctx_mode=video_tef
v_feat_types=video_swin
t_feat_type=bert
results_root=results

######## data paths
train_path=data/ego4d_nlq_moment_detr_train_clip_2s_vl_250.json
eval_path=data/ego4d_nlq_moment_detr_val_clip_2s_vl_250.json
eval_split_name=val

######## setup video+text features
feat_root=/playpen-storage/mmiemon/ego4d/data/v1

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip)
  (( v_feat_dim += 768 ))
fi
if [[ ${v_feat_types} == *"video_swin"* ]]; then
  v_feat_dirs+=(${feat_root}/video_swin)
  (( v_feat_dim += 1024 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text
  t_feat_dim=768
elif [[ ${t_feat_type} == "bert" ]]; then
  t_feat_dir=${feat_root}/bert_text
  t_feat_dim=768
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=64

PYTHONPATH=$PYTHONPATH:. CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python moment_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--num_queries 20 \
--exp_id queries_20_bert_text_video_swin \
${@:1}
