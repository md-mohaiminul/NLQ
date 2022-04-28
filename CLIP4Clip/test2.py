import json
from dataloaders.rawvideo_util import RawVideoExtractor
import cv2
import numpy as np

video_id = 'aae9a55f-866a-44af-bbb1-60ddcab947b7'
video_path = f'/playpen-storage/mmiemon/ego4d/data/v1/clips_fps_3_224/{video_id}.mp4'
# rawVideoExtractor = RawVideoExtractor(framerate=3, size=224)
# raw_video_data = rawVideoExtractor.get_video_data(video_path, 0., 9.1)
#
# print(raw_video_data['video'].shape)

cap = cv2.VideoCapture(video_path)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

total_duration = (frameCount + fps - 1) // fps
start_sec, end_sec = 0, total_duration

print(fps, frameCount)

# if start_time is not None:
#     start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
#     cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
#

cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
interval = 1
sample_fp = 1

if interval == 0: interval = 1

inds = [ind for ind in np.arange(0, fps, interval)]
assert len(inds) >= sample_fp
inds = inds[:sample_fp]
print(inds)


ret = True
images, included = [], []

for sec in np.arange(start_sec, end_sec + 1):
    if not ret: break
    sec_base = int(sec * fps)
    for ind in inds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

cap.release()
#
# if len(images) > 0:
#     video_data = th.tensor(np.stack(images))
# else:
#     video_data = th.zeros(1)
