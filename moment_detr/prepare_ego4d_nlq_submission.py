import json
from utils.basic_utils import load_jsonl
import copy

file = 'results/all_data_mean_pool/ego4d-video_tef-queries_20/best_ego4d_val_preds.jsonl'
result_save_path = 'results/all_data_mean_pool/ego4d-video_tef-queries_20/ego4d_nlq_val.json'
submission = load_jsonl(file)

predictions = []
for record in submission:
    new_datum = {
        "clip_uid": record["vid"],
        "annotation_uid": record['qid'].split('_')[0],
        "query_idx": int(record['qid'].split('_')[1]),
        "predicted_times": copy.deepcopy(record["pred_relevant_windows"]),
    }
    predictions.append(new_datum)

with open(result_save_path, "w") as file_id:
    json.dump(
        {
            "version": "1.0",
            "challenge": "ego4d_nlq_challenge",
            "results": predictions,
        }, file_id
    )

