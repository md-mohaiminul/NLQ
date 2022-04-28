import json
import utils.evaluate_ego4d_nlq as ego4d_eval
import glob

with open('/playpen-storage/mmiemon/ego4d/data/annotations/nlq_val_10s.json') as file_id:
    ground_truth = json.load(file_id)

# with open('checkpoints/nlq_official_clip_10s/vslnet_nlq_official_clip_10s_video_swin_512_bert/model/vslnet_29455_test_result_no_nms.json') as file_id:
#     predictions = json.load(file_id)["results"]

# with open('checkpoints/nlq_official_clip_10s/vslnet_nlq_official_clip_10s_video_swin_512_bert/model/vslnet_29455_test_result.json') as file_id:
#     predictions = json.load(file_id)["results"]
#
# print(len(ground_truth), len(predictions))

predictions =[]
for i in range(3093):
    with open(f'../CLIP4Clip/outputs/vslnet_predictions_clip4clip_nms/{i}.json','r') as f:
        p = json.load(f)
    predictions.append(p)

print(len(predictions))

thresholds = [0.1, 0.3, 0.5]
topK = [1, 5, 10, 20, 50]

# thresholds = [0.1]
# topK = [1]

results, mIoU = ego4d_eval.evaluate_nlq_performance(predictions, ground_truth, thresholds, topK)

score_str = ego4d_eval.display_results(
    results, mIoU, thresholds, topK, title=None
)

print(score_str, flush=True)