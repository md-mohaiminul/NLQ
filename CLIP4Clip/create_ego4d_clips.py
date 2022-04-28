import json
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

root = '/playpen-storage/mmiemon/ego4d/data/v1/full_scale'
data_json = '/playpen-storage/mmiemon/ego4d/data/annotations/nlq_train.json'
with open(data_json, mode="r", encoding="utf-8") as f:
    data = json.load(f)['videos']

for cnt, video_datum in enumerate(data):
    video_file = f'{root}/{video_datum["video_uid"]}.mp4'
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        save_path = f'/playpen-storage/mmiemon/ego4d/data/v1/ego4d_clips/{clip_uid}.mp4'
        ffmpeg_extract_subclip(video_file, clip_datum["video_start_sec"], clip_datum["video_end_sec"],
                               targetname=save_path)
        print(cnt, video_datum["video_uid"], clip_uid)

python preprocess/compress_video.py --input_root /playpen-storage/mmiemon/ego4d/data/v1/ego4d_clips --output_root /playpen-storage/mmiemon/ego4d/data/v1/ego4d_clips_fps_10_224