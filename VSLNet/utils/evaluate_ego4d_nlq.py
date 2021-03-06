#! /usr/bin/env python
"""
Script to evaluate performance of any model for Ego4d Episodic Memory.

Natural Language Queries (NLQ)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import math
import random

import numpy as np
import terminaltables
import torch
from utils.nms import nms, softnms_v2


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


def evaluate_nlq_performance(
    predictions, ground_truth, thresholds, topK, per_instance=False
):
    """Evalutes the performances."""
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

    results = [[[] for _ in topK] for _ in thresholds]
    average_IoU = []
    num_instances = 0

    counts = []
    gt_durations = []
    pred_durations = []
    for pred_datum in predictions:
        key = (pred_datum["clip_uid"], pred_datum["annotation_uid"])
        assert key in gt_dict, "Instance not present!"
        query_id = pred_datum["query_idx"]
        gt_datum = gt_dict[key]
        gt_query_datum = gt_datum["language_queries"][query_id]

        # if 'slot_x' in gt_query_datum and 'slot_y' in gt_query_datum:
        #     continue

        # nms_predictions, count = nms(torch.tensor(pred_datum["predicted_times"]), torch.tensor(pred_datum["scores"]))
        #
        # counts.append(count)

        #Compute overlap and recalls.
        overlap = compute_IoU(
            pred_datum["predicted_times"],
            [[gt_query_datum["clip_start_sec"], gt_query_datum["clip_end_sec"]]],
        )

        # overlap = compute_IoU(
        #     nms_predictions.tolist(),
        #     [[gt_query_datum["clip_start_sec"], gt_query_datum["clip_end_sec"]]],
        # )

        average_IoU.append(np.mean(np.sort(overlap[0])[-3:]))
        for tt, threshold in enumerate(thresholds):
            for rr, KK in enumerate(topK):
                results[tt][rr].append((overlap > threshold)[:KK].any())
                # if (overlap[0] > 0.1) and (overlap[0] < 0.3):
                #     print(gt_query_datum["clip_start_sec"], gt_query_datum["clip_end_sec"],
                #           pred_datum["predicted_times"][0])
                #     gt_durations.append(gt_query_datum["clip_end_sec"]-gt_query_datum["clip_start_sec"])
                #     pred_durations.append(pred_datum["predicted_times"][0][1]-pred_datum["predicted_times"][0][0])
        num_instances += 1

    #print(min(counts), max(counts), sum(counts)/len(counts))
    # print('gt', sum(gt_durations) / len(gt_durations))
    # print('pred', sum(pred_durations) / len(pred_durations))

    mean_results = np.array(results).mean(axis=-1)
    mIoU = np.mean(average_IoU)
    print(f"Evaluated: {num_instances} / {num_gt_queries} instances")
    if per_instance:
        per_instance_results = {
            "overlap": overlap,
            "average_IoU": average_IoU,
            "results": results,
        }
        return mean_results, mIoU, per_instance_results
    else:
        return mean_results, mIoU


def main(args):
    print(f"""Reading predictions: {args["model_prediction_json"]}""")
    with open(args["model_prediction_json"], "r") as file_id:
        predictions = json.load(file_id)

    print(f"""Reading gt: {args["ground_truth_json"]}""")
    with open(args["ground_truth_json"], "r") as file_id:
        ground_truth = json.load(file_id)

    assert predictions.get("version", None) == "1.0", "Ego4D version does not match!"
    assert predictions.get("challenge", None) == "ego4d_nlq_challenge", (
        "Ego4D challenge does not match!"
    )
    results, mIoU = evaluate_nlq_performance(
        predictions["results"], ground_truth, args["thresholds"], args["topK"]
    )
    print(display_results(results, mIoU, args["thresholds"], args["topK"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground_truth_json",
        required=True,
        help="Ground truth temporal windows",
    )
    parser.add_argument(
        "--model_prediction_json",
        required=True,
        help="Model predicted temporal windows",
    )
    parser.add_argument(
        "--thresholds",
        required=True,
        nargs="+",
        type=float,
        help="Thresholds for IoU computation",
    )
    parser.add_argument(
        "--topK",
        required=True,
        nargs="+",
        type=int,
        help="Top K for computing recall@k",
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
