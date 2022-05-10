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

problem = 0
total = 0
for video_datum in split_data["videos"]:
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        for ann_datum in clip_datum["annotations"]:
            annotations_uid = ann_datum["annotation_uid"]
            for i in range(len(ann_datum["language_queries"])):
                x = ann_datum["language_queries"][i]
                ious = []
                cnt = 0
                for j in range(i+1, len(ann_datum["language_queries"])):
                    y = ann_datum["language_queries"][j]
                    iou = compute_IoU([x["clip_start_sec"],x["clip_end_sec"]], [y["clip_start_sec"],y["clip_end_sec"]])
                    if iou>0:
                        cnt += 1
                        ious.append(iou)
                if cnt:
                    print(cnt, len(ann_datum["language_queries"]))
                    problem += 1
                total += 1

print(total, problem)
