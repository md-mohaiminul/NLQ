import json
import torch

clip_features = 'data/features/nlq_official_v1/clip'
read_path = 'data/features/nlq_official_v1/official/feature_shapes.json'
with open(read_path, "r") as file_id:
    data = json.load(file_id)

feature_sizes = {}
for clip_uid in data:
    f = torch.load(f'{clip_features}/{clip_uid}.pt')
    feature_sizes[clip_uid] = f.shape[0]
    print(clip_uid, data[clip_uid], f.shape)

# save_path = f'{clip_features}/feature_shapes.json'
# with open(save_path, "w") as file_id:
#     json.dump(feature_sizes, file_id)