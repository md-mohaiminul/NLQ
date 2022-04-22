import subprocess
import time
import multiprocessing
from multiprocessing import Pool
import shutil
from moviepy.editor import *
import json
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count

from dataloaders.data_loader_ego4d_retrieval import Ego4d_DataLoader
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

# print(cpu_count())
from dataloaders.rawvideo_util import RawVideoExtractor
rawVideoExtractor = RawVideoExtractor(framerate=1, size=224)
raw_video_data = rawVideoExtractor.get_video_data('/playpen-storage/mmiemon/ego4d/data/v1/clips_fps_3_224/aaa29a66-6053-429f-a78f-79272abb34b8.mp4',
                                                       470, 500)
print(raw_video_data['video'].shape)

# split = 'train'
# root = '/playpen-storage/mmiemon/ego4d/data/v1/full_scale'
# data_json = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_{split}.json'
# with open(data_json, mode="r", encoding="utf-8") as f:
#     data = json.load(f)['videos']
#
# cnt = 0
# for video_datum in data:
#     video_file = f'{root}/{video_datum["video_uid"]}.mp4'
#     video = VideoFileClip(video_file)
#     video_duration = video.duration
#     print(video_duration)
#     for clip_datum in video_datum["clips"]:
#         cnt += 1
#         clip_uid = clip_datum["clip_uid"]
#         clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
#         print(clip_uid, clip_duration, video_duration)
#

dataset = Ego4d_DataLoader(subset = 'train',
                           data_path = '/playpen-storage/mmiemon/ego4d/data/annotations/',
                           features_path = '/playpen-storage/mmiemon/ego4d/data/v1/clips_fps_3_224',
                           tokenizer = ClipTokenizer())

print(len(dataset))

pairs_text, pairs_mask, pairs_segment, video, video_mask = dataset.__getitem__(100)

print(pairs_text, len(pairs_text))
print(pairs_mask, len(pairs_mask))
print(video.shape)

print(video_mask)

