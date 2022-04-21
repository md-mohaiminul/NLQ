import json
import os

data_json = '/playpen-storage/mmiemon/ego4d/data/annotations/nlq_val.json'
with open(data_json, mode="r", encoding="utf-8") as f:
    data = json.load(f)['videos']

cnt = 0
for video_datum in data:
    video_uid = video_datum["video_uid"]
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        features = f'/playpen-storage/mmiemon/ego4d/NLQ/VSLNet/data/features/nlq_official_v1/official/{video_uid}.pt'
        if not os.path.exists(features):
            print(cnt, clip_uid)
            cnt += 1
        #print(cnt, clip_uid, features.shape)