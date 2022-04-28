import numpy as np
import torch
import json
import random
import cv2
import sys
import copy

split = 'train'
data_json = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_{split}.json'
with open(data_json, mode="r", encoding="utf-8") as f:
    split_data = json.load(f)

all_len = []
cnt = 0
missing = 0
for video_datum in split_data["videos"]:
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        for ann_datum in clip_datum["annotations"]:
            annotations_uid = ann_datum["annotation_uid"]
            for index, datum in enumerate(copy.deepcopy(ann_datum["language_queries"])):
                if "query" not in datum or not datum["query"]:
                    ann_datum["language_queries"].remove(datum)
                    continue
                duration = datum["clip_end_sec"] - datum["clip_start_sec"]
                if duration > 10:
                    missing += 1
                    ann_datum["language_queries"].remove(datum)
                quid = annotations_uid + '_' + str(index)
                cnt += 1

print(cnt, missing, cnt-missing)

cnt = 0
durations = []
mids = []
for video_datum in split_data["videos"]:
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
        for ann_datum in clip_datum["annotations"]:
            annotations_uid = ann_datum["annotation_uid"]
            for index, datum in enumerate(ann_datum["language_queries"]):
                duration = datum["clip_end_sec"] - datum["clip_start_sec"]
                if duration <2 :
                    mid = min(max(1, (datum["clip_end_sec"] + datum["clip_start_sec"])/2), clip_duration-1)

                    ann_datum["language_queries"][index]["clip_start_sec"] =  mid -1
                    ann_datum["language_queries"][index]["clip_end_sec"] = mid + 1
                    mids.append(mid)
                durations.append(duration)
                cnt += 1

print('mid', min(mids), max(mids))

cnt = 0
durations = []
for video_datum in split_data["videos"]:
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        for ann_datum in clip_datum["annotations"]:
            annotations_uid = ann_datum["annotation_uid"]
            for index, datum in enumerate(ann_datum["language_queries"]):
                duration = datum["clip_end_sec"] - datum["clip_start_sec"]
                durations.append(duration)
                cnt += 1

print(min(durations), max(durations))
print(cnt)

save_path = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_{split}_10s.json'
with open(save_path, "w") as file_id:
    json.dump(split_data, file_id)

