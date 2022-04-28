import json

with open('checkpoints/nlq_official_clip_10s/vslnet_nlq_official_clip_10s_video_swin_512_bert/model/vslnet_29455_test_result.json') as file_id:
    predictions = json.load(file_id)["results"]

print(predictions.keys())