import json
import math

import clip
import torch
import numpy as np

with open('/playpen-storage/mmiemon/ego4d/data/annotations/nlq_train.json') as file_id:
    ground_truth = json.load(file_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)

ll = []
key = 'slot_y'

for video_datum in ground_truth["videos"]:
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        for ann_datum in clip_datum["annotations"]:
            for index, datum in enumerate(ann_datum["language_queries"]):
                if "query" not in datum or not datum["query"]:
                    continue
                if key in datum and datum[key]:
                    ll.append(datum[key])

print(len(ll))
ll = list(set(ll))
print(len(ll))

softmax = torch.nn.Softmax(dim=1)

text_features = []
for cnt, x in enumerate(ll):
    with torch.no_grad():
        text_inputs = clip.tokenize(x).to(device)
        tf = model.encode_text(text_inputs).float().detach().cpu() #[1, 768]
        tf /= tf.norm(dim=-1, keepdim=True)
        text_features.append(tf)

text_features = torch.cat(text_features, dim=0)
print(text_features.shape)

top_k = [1, 5, 10, 20, 50]
result = {}
for k in top_k:
    result[k] = 0
total = 0
for video_datum in ground_truth["videos"]:
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        features = f'/playpen-storage/mmiemon/ego4d/NLQ/VSLNet/data/features/nlq_official_v1/clip/{clip_uid}.pt'
        features = torch.load(features)
        idx = [i for i in range(features.shape[0]) if i % 10 == 0]
        features = features[idx]
        for ann_datum in clip_datum["annotations"]:
            for index, datum in enumerate(ann_datum["language_queries"]):
                if "query" not in datum or not datum["query"]:
                    continue
                if key in datum and datum[key]:
                    s = math.floor(datum['clip_start_sec']*3)
                    e = math.floor(datum['clip_end_sec']*3)
                    image_features = features[s:e+1]
                    # image_features = torch.unsqueeze(torch.mean(image_features, dim=0), dim=0)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    sim_matrix = (image_features @ text_features.T)
                    sim_matrix = softmax(sim_matrix)
                    sim_matrix = torch.mean(sim_matrix, dim=0)
                    idx = np.argsort(-sim_matrix)
                    for k in top_k:
                        if ll.index(datum[key]) in idx[:k]:
                            result[k] += 1
                    total += 1

for k in result:
    result[k] = (result[k]/ total)*100.0

print(result)

# validation set:
# total x: 3651, unique x: 2110
# classification x topk: {1: 1.42, 5: 5.94, 10: 9.69, 20: 14.51, 50: 22.78}
# total y: 1257, unique y: 724
# classification y topk: {1: 2.54, 5: 7.15, 10: 9.78, 20: 13.36, 50: 20.36}
#
# train set:
# total x: 10778, unique x: 4979
# classification x topk: {1: 0.60, 5: 2.72, 10: 5.01, 20: 8.05, 50: 13.98}
# total y: 3458, unique y: 1934
# classification y topk: {1: 1.21, 5: 3.87, 10: 6.36, 20: 9.54, 50: 15.58}


