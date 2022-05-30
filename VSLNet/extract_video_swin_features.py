import json
import os
from moviepy.editor import *
import numpy as np
import torch
import torch.nn as nn
import cv2
import math
import random
from mmaction.apis import init_recognizer, inference_recognizer
from mmaction.models import build_model

stride = 16
window = 32
fps = 30.0

config_file = '/playpen-storage/mmiemon/lvu_state_space/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
checkpoint_file = '/playpen-storage/mmiemon/lvu_state_space/Video-Swin-Transformer/checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = init_recognizer(config_file, checkpoint_file, device=device)
avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)).to(device)

root = '/playpen-storage/mmiemon/ego4d/data/v1/full_scale'
data_json = '/playpen-storage/mmiemon/ego4d/Ego4d/annotations/v1/annotations/nlq_test_unannotated.json'
with open(data_json, mode="r", encoding="utf-8") as f:
    data = json.load(f)['videos']

random.shuffle(data)

print(len(data))

cnt = 0
for video_datum in data:
    video_file = f'{root}/{video_datum["video_uid"]}.mp4'
    video = VideoFileClip(video_file)
    print(video_file)
    for clip_datum in video_datum["clips"]:
        clip_uid = clip_datum["clip_uid"]
        cnt += 1
        save_path = f'/playpen-storage/mmiemon/ego4d/data/v1/video_swin/{clip_uid}.pt'
        if os.path.exists(save_path):
            continue
        clip_uid = clip_datum["clip_uid"]
        clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
        print(clip_uid, clip_datum["video_start_sec"], clip_datum["video_end_sec"], clip_duration)
        n_frames = int(clip_duration*fps)

        all_features = torch.rand([0,1024])
        for start in range(0, n_frames - window + 1, stride):
            frames = []
            for i in range(start, start + window):
                t = clip_datum["video_start_sec"] + (i / fps)
                image = video.get_frame(clip_datum["video_start_sec"] + (i / fps))
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                frames.append(image)

            frames = np.asarray(frames) / 255.0
            frames = torch.from_numpy(frames.transpose([3, 0, 1, 2])).float()
            frames = torch.unsqueeze(frames, 0)

            features = model.extract_feat(frames.to(device))
            features = avg_pool(features).view(features.shape[0], -1).detach().cpu()
            all_features = torch.cat((all_features, features), 0)

            print('Clip ', clip_uid, cnt, '/', 333, ' frame start ', start, '/', n_frames)

        torch.save(all_features, save_path)

        print(clip_uid, ' duration ', clip_duration, ' frames ', n_frames, ' features ', all_features.shape)
