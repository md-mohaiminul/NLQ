import json
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
from moviepy.editor import VideoFileClip

root = '/playpen-storage/mmiemon/ego4d/data/v1/full_scale'
json_file = '/playpen-storage/mmiemon/ego4d/Ego4d/annotations/v1/annotations/fho_oscc-pnr_val.json'
with open(json_file, 'r') as f:
    data = json.load(f)
print(data['split'])
data = data['clips']
print(len(data))

durations = []
for cnt, item in enumerate(data):
    video_uid = item['video_uid']
    clip_uid = item['unique_id']
    durations.append(item['parent_end_sec'] - item['parent_start_sec'])
    clip_file = f'/playpen-storage/mmiemon/ego4d/data/v1/fho_clips/{clip_uid}.mp4'
    #clip_file = 'test.mp4'
    clip = VideoFileClip(clip_file)
    print(clip.fps)
    print(video_uid, clip_uid, clip.duration, item['parent_end_sec'], item['parent_start_sec'])
    break
