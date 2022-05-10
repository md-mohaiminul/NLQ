import json
import math
import random

import numpy as np
import terminaltables
import torch

from itertools import compress

def display_results(results, mIoU, thresholds, topK, title=None):
    display_data = [
        [f"Rank@{ii}\nmIoU@{jj}" for ii in topK for jj in thresholds] + ["mIoU"]
    ]
    results *= 100
    mIoU *= 100
    display_data.append(
        [
            f"{results[jj][ii]:.02f}"
            for ii in range(len(topK))
            for jj in range(len(thresholds))
        ]
        + [f"{mIoU:.02f}"]
    )
    table = terminaltables.AsciiTable(display_data, title)
    for ii in range(len(thresholds) * len(topK)):
        table.justify_columns[ii] = "center"
    return table.table


def compute_IoU(pred, gt):
    """Compute the IoU given predicted and ground truth windows."""
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)

    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]

    # if gt[0][0]==gt[0][1]:
    #     print(overlap)

    return overlap

with open('/playpen-storage/mmiemon/ego4d/data/annotations/nlq_train_10s.json') as file_id:
    ground_truth = json.load(file_id)

with open('checkpoints/nlq_official_clip_10s/vslnet_nlq_official_clip_10s_video_swin_512_bert/model/vslnet_29455_train_no_nms.json') as file_id:
    predictions = json.load(file_id)["results"]

gt_dict = {}
num_gt_queries = 0

for video_datum in ground_truth["videos"]:
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
        for ann_datum in clip_datum["annotations"]:
            key = (clip_uid, ann_datum["annotation_uid"])
            gt_dict[key] = ann_datum
            num_gt_queries += len(ann_datum["language_queries"])

counts = []
gt_durations = []
pred_durations = []
lens = []
for cnt, pred_datum in enumerate(predictions):
    key = (pred_datum["clip_uid"], pred_datum["annotation_uid"])
    assert key in gt_dict, "Instance not present!"
    query_id = pred_datum["query_idx"]
    gt_datum = gt_dict[key]
    gt_query_datum = gt_datum["language_queries"][query_id]

    # print(pred_datum.keys())
    #Compute overlap and recalls.
    overlap = compute_IoU(
        pred_datum["predicted_times"],
        [[gt_query_datum["clip_start_sec"], gt_query_datum["clip_end_sec"]]],
    )

    new_pred = []
    for i in range(len(overlap)):
        if len(new_pred)>= 50:
            break
        x = pred_datum["predicted_times"][i]
        if (overlap[i]<0.1) and (x[1]-x[0])<=50 and (x not in new_pred):
            new_pred.append(x)
    lens.append(len(new_pred))
    print(cnt, len(new_pred))

    #pred_datum["predicted_times"] = new_pred

    gt_query_datum['negatives'] = new_pred

# print(min(lens), max(lens), sum(lens)/len(lens))
#
# d = []
# for cnt, pred_datum in enumerate(predictions):
#     for t in pred_datum["predicted_times"]:
#         d.append(t[1]-t[0])
#
# print(min(d), max(d), sum(d)/len(d))
#
# save_path = 'checkpoints/nlq_official_clip_10s/vslnet_nlq_official_clip_10s_video_swin_512_bert/model/vslnet_29455_train_negatives.json'
# with open(save_path, "w") as file_id:
#     json.dump(predictions, file_id)

d = []
for video_datum in ground_truth["videos"]:
    for clip_datum in video_datum["clips"]:
        for ann_datum in clip_datum["annotations"]:
            for gt_query_datum in ann_datum["language_queries"]:
                for t in gt_query_datum['negatives']:
                    d.append(t[1] - t[0])

print(min(d), max(d), sum(d)/len(d))

save_path = '/playpen-storage/mmiemon/ego4d/data/annotations/nlq_train_neg_10s.json'
with open(save_path, "w") as file_id:
    json.dump(ground_truth, file_id)



