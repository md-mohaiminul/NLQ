from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import math
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
from dataloaders.rawvideo_util import RawVideoExtractor
import multiprocessing

def process_question(question):
    """Process the question to make it canonical."""
    return question.strip(" ").strip("?").lower() + "?"

class PredictionDataset(Dataset):
    def __init__(self, choice_video_ids, negative_pool, _get_rawvideo):
        self.negative_pool = negative_pool
        self._get_rawvideo = _get_rawvideo
        self.choice_video_ids = choice_video_ids

    def __len__(self):
        return len(self.negative_pool)

    def __getitem__(self, i):
        neg = random.choice(self.negative_pool)
        video, video_mask = self._get_rawvideo(self.choice_video_ids, math.floor(neg[0]), math.ceil(neg[1]))
        return video, video_mask

class Ego4d_DataLoader_same_video(Dataset):
    """Ego4d dataset loader."""
    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "val", "test"]

        annotation_dict = {}
        annotation_dict["train"] = os.path.join(self.data_path, "nlq_train_neg_10s.json")
        annotation_dict["val"] = os.path.join(self.data_path, "nlq_val_10s.json")
        annotation_dict["test"] = os.path.join(self.data_path, "nlq_val_10s.json")

        with open(annotation_dict[self.subset], mode="r", encoding="utf-8") as f:
            split_data = json.load(f)

        self.item_dict = {}
        self.sentences_dict = {}
        self.video_dict = {}
        for video_datum in split_data["videos"]:
            for clip_datum in video_datum["clips"]:
                clip_uid = clip_datum["clip_uid"]
                self.video_dict[clip_uid] = os.path.join(self.features_path, f'{clip_uid}.mp4')
                for ann_datum in clip_datum["annotations"]:
                    for index, datum in enumerate(ann_datum["language_queries"]):
                        if "query" not in datum or not datum["query"]:
                            continue
                        new_dict = {
                            "video_id": clip_uid,
                            "start_time": math.floor(datum["clip_start_sec"]),
                            "end_time": math.ceil(datum["clip_end_sec"]),
                            "sentence": process_question(datum["query"]),
                            "negatives": datum['negatives'],
                        }

                        self.sentences_dict[len(self.sentences_dict)] = new_dict

        print("Total videos: {}".format(len(self.video_dict)))
        print("Total Pairs: {}".format(len(self.sentences_dict)))

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.sentences_dict)

    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids, start_time, end_time):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        item = self.sentences_dict[idx]
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(item["video_id"], item["sentence"])
        video, video_mask = self._get_rawvideo(choice_video_ids, item["start_time"], item["end_time"])
        negative_videos = []
        negative_video_masks = []
        negatives = random.sample(item['negatives'], 10)
        for neg in negatives:
            s = math.floor(neg[0])
            e = math.ceil(neg[1])
            vn, vmn = self._get_rawvideo(choice_video_ids, s, min(s+50, e))
            negative_videos.append(vn)
            negative_video_masks.append(vmn)
        negative_videos = np.stack(negative_videos, axis=0)
        negative_video_masks = np.stack(negative_video_masks, axis=0)

        # predictionDataset = PredictionDataset(choice_video_ids, item['negatives'], self._get_rawvideo)
        # prediction_loader = DataLoader(
        #     predictionDataset,
        #     batch_size=10,
        #     num_workers=10,
        #     pin_memory=False,
        #     shuffle=False,
        # )
        # negative_batch, negative_mask_batch = next(iter(prediction_loader))
        #
        # print(negative_batch.shape, negative_mask_batch.shape)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, negative_videos, negative_video_masks
