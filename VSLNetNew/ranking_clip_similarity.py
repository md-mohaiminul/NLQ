import json
import clip
import torch
import math
import numpy as np

def process_question(question):
    """Process the question to make it canonical."""
    return question.strip(" ").strip("?").lower() + "?"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device, jit=False)

with open('checkpoints/clip_128_512_nce/model/vslnet_35200_test_result.json') as file_id:
    json_dict = json.load(file_id)

with open('/playpen-storage/mmiemon/ego4d/Ego4d/annotations/v1/annotations/nlq_val.json') as file_id:
    ground_truth = json.load(file_id)

predictions = json_dict["results"]

gt_dict = {}
for video_datum in ground_truth["videos"]:
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        for ann_datum in clip_datum["annotations"]:
            key = (clip_uid, ann_datum["annotation_uid"])
            gt_dict[key] = ann_datum

topk = 20
fps = 30
for cnt, pred_datum in enumerate(predictions):
    key = (pred_datum["clip_uid"], pred_datum["annotation_uid"])
    assert key in gt_dict, "Instance not present!"
    query_id = pred_datum["query_idx"]
    gt_datum = gt_dict[key]
    gt_query_datum = gt_datum["language_queries"][query_id]

    clip_uid = pred_datum['clip_uid']
    features = f'/playpen-storage/mmiemon/ego4d/data/v1/clip/{clip_uid}.pt'
    features = torch.load(features)

    # idx = [i for i in range(features.shape[0]) if i%10==0]
    # features = features[idx]

    text = process_question(gt_query_datum['query'])

    text_inputs = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)['pooler_output']
        text_features = text_features.float().detach().cpu() #[1, 768]
    print(text_features.shape, len(pred_datum['predicted_times']))

    image_features = torch.zeros([topk, 768])
    for i in range(topk):
        s = math.floor(pred_datum['predicted_times'][i][0]*fps)
        e = math.ceil(pred_datum['predicted_times'][i][1]*fps)
        f = torch.mean(features[s:e+1], dim=0)
        image_features[i] = f

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    sim_matrix = (text_features @ image_features.T)[0]
    idx = np.argsort(-sim_matrix)

    pred_datum['predicted_times'] = np.asarray(pred_datum['predicted_times'])
    pred_datum['predicted_times'][:topk] = pred_datum['predicted_times'][:topk][idx]
    pred_datum['predicted_times'] = pred_datum['predicted_times'].tolist()
    print(cnt, '/', len(predictions))

with open(f'checkpoints/clip_128_512_nce/model/vslnet_35200_val_result_clip_similarity.json', "w") as file_id:
    json.dump(json_dict, file_id)