import glob
import os
import json

from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

source = '/playpen-storage/mmiemon/ego4d/data/v1/full_scale_fps_3_224'
destination = '/playpen-storage/mmiemon/ego4d/data/v1/clips_fps_3_224'

split = 'val'
data_json = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_{split}.json'
with open(data_json, mode="r", encoding="utf-8") as f:
    data = json.load(f)['videos']

cnt = 0
for video_datum in data:
    video_file = f'{source}/{video_datum["video_uid"]}.mp4'
    for clip_datum in video_datum["clips"]:
        cnt += 1
        clip_uid = clip_datum["clip_uid"]
        clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
        ffmpeg_extract_subclip(video_file, clip_datum["video_start_sec"], clip_datum["video_end_sec"],
                               targetname=f"{destination}/{clip_uid}.mp4")
        print(clip_uid, clip_uid, clip_duration)