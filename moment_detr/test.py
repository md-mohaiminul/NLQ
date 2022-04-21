from moment_detr.start_end_dataset_ego4d import StartEndDataset
import torch
import json
from utils.basic_utils import load_jsonl

dataset = StartEndDataset("ego4d", 'data/ego4d_nlq_moment_detr_train_clip_2s_vl_250.json',
                           ['/playpen-storage/mmiemon/ego4d/data/v1/clip'],
                           '/playpen-storage/mmiemon/ego4d/data/v1/clip_text')

print(len(dataset))

dataset.__getitem__(0)

# file = 'results/query_ablation/ego4d-video_tef-queries_10/best_ego4d_val_preds.jsonl'
# # with open('results/query_ablation/ego4d-video_tef-queries_10/best_ego4d_val_preds.json', 'r') as f:
# #     submission = json.load(f)
# submission = load_jsonl(file)
# for item in submission:
#     print(item.keys())
#     print(item['pred_relevant_windows'])
#     break



