import os
import json
import torch
import random
#
# json_file = '/playpen-storage/mmiemon/ego4d/data/annotations/fho_oscc-pnr_train.json'
#
# with open(json_file, 'r') as f:
#     data = json.load(f)
#
# print(data['split'])
#
# data = data['clips']
#
# print(len(data))
#
# for item in data:
#     for k in item:
#         print(k, item[k])
#     # video_uid = item['video_uid']
#     # if not os.path.exists(f'/playpen-storage/mmiemon/ego4d/data/v1/full_scale/{video_uid}.mp4'):
#     #     print(video_uid)

indices = random.sample(range(10, 20), 3)
indices.sort()
print(indices)