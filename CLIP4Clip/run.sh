python -m torch.distributed.launch --nproc_per_node=8 \
main_task_retrieval.py --do_train --num_thread_reader=48 \
--epochs=10 --batch_size=128 --n_display=50 \
--data_path /playpen-storage/mmiemon/ego4d/data/annotations/ \
--features_path /playpen-storage/mmiemon/ego4d/data/v1/ego4d_clips_fps_10_224 \
--output_dir ckpts/ViT_B_16_fps_10/ \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 128 \
--datatype ego4d \
--feature_framerate 10 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16

#python preprocess/compress_video.py --input_root [raw_video_path] --output_root [compressed_video_path]