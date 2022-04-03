import json
import torch

swin_features = '/playpen-storage/mmiemon/ego4d/data/v1/video_swin'
read_path = 'data/features/nlq_official_v1/official/feature_shapes.json'
with open(read_path, "r") as file_id:
    data = json.load(file_id)

feature_sizes = {}
for clip_uid in data:
    f = torch.load(f'{swin_features}/{clip_uid}.pt')
    feature_sizes[clip_uid] = f.shape[0]
    print(clip_uid, data[clip_uid], f.shape[0])

save_path = '/playpen-storage/mmiemon/ego4d/data/v1/video_swin/feature_shapes.json'
with open(save_path, "w") as file_id:
    json.dump(feature_sizes, file_id)