import json
from dataloaders.rawvideo_util import RawVideoExtractor
import cv2
import numpy as np
import timeit


video_id = 'aae9a55f-866a-44af-bbb1-60ddcab947b7'
#video_id = 'c2cc6ea2-eb70-4524-a69a-363c485e4e03'
video_path = f'/playpen-storage/mmiemon/ego4d/data/v1/ego4d_clips_fps_10_224/{video_id}.mp4'

# cap = cv2.VideoCapture(video_path)
# frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))

rawVideoExtractor = RawVideoExtractor(framerate=10, size=224)

start = timeit.default_timer()
raw_video_data = rawVideoExtractor.get_video_data(video_path, 0, 1)
stop = timeit.default_timer()
print('Time: ', stop - start)

print(raw_video_data['video'].shape)
