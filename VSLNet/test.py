import json
import torch

slowfast = 'data/features/nlq_official_v1/official/feature_shapes.json'
swin = '/playpen-storage/mmiemon/ego4d/data/v1/video_swin/feature_shapes.json'
with open(slowfast, "r") as file_id:
    slowfast = json.load(file_id)

with open(swin, "r") as file_id:
    swin = json.load(file_id)

for clip_uid in slowfast:
    print(clip_uid, slowfast[clip_uid], swin[clip_uid])