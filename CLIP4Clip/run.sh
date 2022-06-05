python -m torch.distributed.launch --nproc_per_node=8 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=16 --n_display=50 \
--data_path /playpen-storage/mmiemon/ego4d/data/annotations/ \
--features_path /playpen-storage/mmiemon/ego4d/data/v1/full_scale_fps_3 \
--output_dir ckpts/same_video/ \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype ego4d \
--feature_framerate 3 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16


#python -m torch.distributed.launch --nproc_per_node=8 \
#main_task_retrieval.py --do_train --num_thread_reader=8 \
#--epochs=5 --batch_size=32 --n_display=50 \
#--data_path /playpen-storage/mmiemon/ego4d/data/annotations/ \
#--features_path /playpen-storage/mmiemon/ego4d/data/v1/clips_fps_3_224 \
#--output_dir ckpts/activity_ViT-B-16 \
#--lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 32 \
#--datatype ego4d --feature_framerate 3 --coef_lr 1e-3 \
#--freeze_layer_num 0  --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--pretrained_clip_name ViT-B/16
#python preprocess/compress_video.py --inp[ut_root [raw_video_path] --output_root [compressed_video_path]

#DATA_PATH=/playpen-storage/mmiemon/MSVD
#python -m torch.distributed.launch --nproc_per_node=4 \
#main_task_retrieval.py --do_train --num_thread_reader=2 \
#--epochs=5 --batch_size=32 --n_display=50 \
#--data_path ${DATA_PATH}/msvd_data \
#--features_path ${DATA_PATH}/YouTubeClips \
#--output_dir ckpts/ckpt_msvd_retrieval_looseType \
#--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
#--datatype msvd \
#--feature_framerate 1 --coef_lr 1e-3 \
#--freeze_layer_num 0 --slice_framepos 2 \
#--loose_type --linear_patch 2d --sim_header meanP \
#--pretrained_clip_name ViT-B/32