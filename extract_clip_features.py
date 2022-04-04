import numpy as np
import torch
import clip
import json
import random
import cv2
from moviepy.editor import *
import sys

from clip_preprocessing import Preprocessing

if int(sys.argv[1])==0:
    split = 'train'
    total_clips = 998
else:
    split = 'val'
    total_clips = 328

fps = 30.0
batch_size = int(sys.argv[2])
print(split, batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess = clip.load('ViT-L/14')
input_preprocess = Preprocessing()
model.to(device).eval()

root = '/playpen-storage/mmiemon/ego4d/data/v1/full_scale'
data_json = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_{split}.json'
with open(data_json, mode="r", encoding="utf-8") as f:
    data = json.load(f)['videos']

random.shuffle(data)

cnt = 0
for video_datum in data:
    video_file = f'{root}/{video_datum["video_uid"]}.mp4'
    video = VideoFileClip(video_file)
    print(video_file)
    for clip_datum in video_datum["clips"]:
        cnt += 1
        clip_uid = clip_datum["clip_uid"]
        save_path = f'/playpen-storage/mmiemon/ego4d/NLQ/VSLNet/data/features/nlq_official_v1/clip/{clip_uid}.pt'
        if os.path.exists(save_path):
            continue

        clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
        n_frames = int(clip_duration*fps)
        print(clip_uid, clip_datum["video_start_sec"], clip_datum["video_end_sec"], clip_duration, n_frames)

        all_features = torch.rand([0,768])
        for start in range(0, n_frames, batch_size):
            frames = []
            for i in range(start, min(start + batch_size, n_frames)):
                t = clip_datum["video_start_sec"] + (i / fps)
                image = video.get_frame(clip_datum["video_start_sec"] + (i / fps))
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                frames.append(image)

            frames = np.asarray(frames)
            frames = torch.from_numpy(frames.transpose([0, 3, 1, 2])).float()
            frames = input_preprocess(frames).to(device)

            with torch.no_grad():
                features = model.encode_image(frames).float().detach().cpu()
                all_features = torch.cat((all_features, features), 0)

            print('Clip ', clip_uid, cnt, '/', total_clips, ' frame start ', start, '/', n_frames, all_features.shape)

        torch.save(all_features, save_path)