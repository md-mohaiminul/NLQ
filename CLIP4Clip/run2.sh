python -m torch.distributed.launch --nproc_per_node=4 \
main_task_retrieval.py --do_eval --num_thread_reader=16 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path /playpen-storage/mmiemon/ego4d/data/annotations/ \
--features_path /playpen-storage/mmiemon/ego4d/data/v1/clips_fps_3_224 \
--output_dir ckpts/ViT_B_16 \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype ego4d  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model ckpts/ViT_B_16/pytorch_model.bin.2 \
--pretrained_clip_name ViT-B/16