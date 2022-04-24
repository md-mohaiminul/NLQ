import numpy as np
import torch
import json
import random
import cv2
import sys
import copy
from utils.nms import nms

x = torch.randn([2, 5])
print(x)
indices = torch.tensor([[1,2],[3,4]])
# print(indices.shape)
y = torch.zeros([2,2])
for i in range(indices.shape[0]):
    y[i] = x[i][indices[i]]
print(y)
y = y.tolist()
print(y)

# split = 'train'
# data_json = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_{split}_10s.json'
# with open(data_json, mode="r", encoding="utf-8") as f:
#     split_data = json.load(f)
#
# all_len = []
# cnt = 0
# missing = 0
# for video_datum in split_data["videos"]:
#     for clip_datum in video_datum["clips"]:
#         clip_uid = clip_datum["clip_uid"]
#         clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
#         for ann_datum in clip_datum["annotations"]:
#             annotations_uid = ann_datum["annotation_uid"]
#             for index, datum in enumerate(copy.deepcopy(ann_datum["language_queries"])):
#                 duration = datum["clip_end_sec"] - datum["clip_start_sec"]
#                 if duration > 10 or duration < 1.99:
#                     missing += 1
#                     ann_datum["language_queries"].remove(datum)
#                 if datum["clip_start_sec"] < 0:
#                     print("error")
#                 if datum["clip_end_sec"] > clip_duration:
#                     print(datum["clip_start_sec"], datum["clip_end_sec"], clip_duration)
#                 quid = annotations_uid + '_' + str(index)
#                 cnt += 1
#
# print(cnt, missing, cnt-missing)