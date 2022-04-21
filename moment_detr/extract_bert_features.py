import torch
from transformers import BertTokenizer, BertForPreTraining
from transformers import BertModel, BertConfig
import json
import numpy as np

device = "cuda"

def pad_seq(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(device)

# word_ids = [tokenizer("how are you?")]
# print(word_ids)
#
# pad_input_ids, _ = pad_seq([ii["input_ids"] for ii in word_ids])
# pad_attention_mask, _ = pad_seq([ii["attention_mask"] for ii in word_ids])
# pad_token_type_ids, _ = pad_seq([ii["token_type_ids"] for ii in word_ids])
# word_ids = {
#     "input_ids": torch.LongTensor(pad_input_ids),
#     "attention_mask": torch.LongTensor(pad_attention_mask),
#     "token_type_ids": torch.LongTensor(pad_token_type_ids),
# }
# outputs = model(**word_ids)
# print(outputs.keys())
# features = outputs["last_hidden_state"].detach()
# print(features.shape)


def process_question(question):
    """Process the question to make it canonical."""
    return question.strip(" ").strip("?").lower() + "?"

root = '/playpen-storage/mmiemon/ego4d/data/v1/bert_text'
data_json = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_val.json'
with open(data_json, mode="r", encoding="utf-8") as f:
    split_data = json.load(f)

all_ann = []
clips = []
cnt = 0
all_len = []
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
                word_ids = [tokenizer(query)]
                pad_input_ids, _ = pad_seq([ii["input_ids"] for ii in word_ids])
                pad_attention_mask, _ = pad_seq([ii["attention_mask"] for ii in word_ids])
                pad_token_type_ids, _ = pad_seq([ii["token_type_ids"] for ii in word_ids])
                word_ids = {
                    "input_ids": torch.LongTensor(pad_input_ids).to(device),
                    "attention_mask": torch.LongTensor(pad_attention_mask).to(device),
                    "token_type_ids": torch.LongTensor(pad_token_type_ids).to(device),
                }
                outputs = model(**word_ids)
                query_feats = outputs["last_hidden_state"][0].detach().cpu().numpy()
                print(cnt, query, query_feats.shape)

                all_len.append(query_feats.shape[0])
                # save_path = f'{root}/{quid}.npy'
                # np.save(save_path, query_feats)
                cnt += 1

print(min(all_len), max(all_len))

print(len(all_ann), len(clips))