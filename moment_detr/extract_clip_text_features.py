from run_on_video.data_utils import ClipFeatureExtractor
import json
import numpy as np

root = '/playpen-storage/mmiemon/ego4d/data/v1/clip_text_clip_10s'

device = "cuda"

feature_extractor = ClipFeatureExtractor(framerate=1/2, size=224, centercrop=True,model_name_or_path="ViT-L/14", device=device)

def process_question(question):
    """Process the question to make it canonical."""
    return question.strip(" ").strip("?").lower() + "?"

data_json = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_train_10s.json'
with open(data_json, mode="r", encoding="utf-8") as f:
    split_data = json.load(f)

all_ann = []
clips = []
cnt = 0
for video_datum in split_data["videos"]:
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        clips.append(clip_uid)
        for ann_datum in clip_datum["annotations"]:
            annotations_uid = ann_datum["annotation_uid"]
            all_ann.append(annotations_uid)
            for index, datum in enumerate(ann_datum["language_queries"]):
                if "query" not in datum or not datum["query"]:
                    continue
                quid = annotations_uid + '_' + str(index)
                query = process_question(datum["query"])
                query_feats = feature_extractor.encode_text(query)[0].detach().cpu().numpy()
                print(cnt, quid, query_feats.shape)
                save_path = f'{root}/{quid}.npy'
                np.save(save_path, query_feats)
                cnt += 1

print(len(all_ann), len(clips))
print(cnt)