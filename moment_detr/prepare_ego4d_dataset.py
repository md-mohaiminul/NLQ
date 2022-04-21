import json
import math

import numpy as np
import torch

from utils.data_utils import index_to_time, time_to_index

def process_question(question):
    """Process the question to make it canonical."""
    return question.strip(" ").strip("?").lower() + "?"

split = 'train'
data_json = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_{split}.json'
with open(data_json, mode="r", encoding="utf-8") as f:
    split_data = json.load(f)

with open('/playpen-storage/mmiemon/ego4d/data/v1/clip/feature_shapes.json') as json_file:
    feature_shape = json.load(json_file)

lengths = []
ideal = 0

cnt = 0
data = []
for video_datum in split_data["videos"]:
    video = video_datum['video_uid']
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
        if clip_duration>=500:
            continue
        for ann_datum in clip_datum["annotations"]:
            annotations_uid = ann_datum["annotation_uid"]
            for index, datum in enumerate(ann_datum["language_queries"]):
                if "query" not in datum or not datum["query"]:
                    continue
                qid = annotations_uid + '_' + str(index)
                query = process_question(datum["query"])
                item = {'qid': qid}
                item['query'] = query
                item['duration'] = clip_duration
                item['vid'] = clip_uid
                item['relevant_windows'] = [[datum['clip_start_sec'], datum['clip_end_sec']]]
                # s, e, _ = time_to_index(item["relevant_windows"][0][0], item["relevant_windows"][0][1]
                #                         , min(feature_shape[clip_uid], 500), item['duration'])
                # s = min(499, math.floor(datum['clip_start_sec']/clip_duration*500))
                # e = min(499, math.ceil(datum['clip_end_sec'] / clip_duration * 500))
                #item["relevant_clip_ids"] = [i for i in range(s, e + 1)]

                s = min(249, int(item["relevant_windows"][0][0] / 2))
                e = min(250, max(s + 1, int(item["relevant_windows"][0][1] / 2)))
                item['relevant_clip_ids'] = [i for i in range(s, e)]
                item['saliency_scores'] = [[1, 1, 1] for i in range(s, e)]
                data.append(item)
                cnt += 1
                print(cnt, s, e, datum['clip_start_sec'], datum['clip_end_sec'], clip_duration)

                lengths.append(datum['clip_end_sec'] - datum['clip_start_sec'])
                if 30<= (datum['clip_end_sec'] - datum['clip_start_sec']) < 1000:
                    ideal += 1

print(min(lengths), max(lengths), sum(lengths)/len(lengths))
print(cnt, ideal)
# print(split, cnt)
#
# json_string = json.dumps(data)
#
# with open(f"data/ego4d_nlq_moment_detr_{split}_clip_2s_vl_250.json", "w") as outfile:
#     outfile.write(json_string)


