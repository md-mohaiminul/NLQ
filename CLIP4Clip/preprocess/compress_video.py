"""
Used to compress video in: https://github.com/ArrowLuo/CLIP4Clip
Author: ArrowLuo
"""
import glob
import os
import argparse
import ffmpeg
import subprocess
import time
import multiprocessing
from multiprocessing import Pool
import shutil
import json
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
# multiprocessing.freeze_support()

# def compress(paras):
#     input_video_path, output_video_path = paras
#     output_video_path_ori = output_video_path  # .split('.')[0]+'.mp4'
#
#     output_video_path = os.path.splitext(output_video_path)[0]  # +'.mp4'
#
#     if not os.path.exists(output_video_path):
#         os.makedirs(output_video_path)
#
#     ## for audio/images extractation
#     output_img_path = output_video_path + '/%04d.jpg'
#     output_audio_path = output_video_path + '/%04d.wav'
#     # output_audio2_path = output_video_path.split('.')[0]+'.wav'
#     output_audio2_path = output_video_path + '.wav'
#     try:
#         # command = ['ffmpeg',
#         #     '-y',  # (optional) overwrite output file if it exists
#         #     '-i', input_video_path,
#         #     '-c:v',
#         #     'libx264',
#         #     '-c:a',
#         #     'libmp3lame',
#         #     '-b:a',
#         #     '128K',
#         #     '-max_muxing_queue_size', '9999',
#         # 	'-vf',
#         #     'fps=3 ',  # scale to 224 "scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'"
#         #     # '-max_muxing_queue_size', '9999',
#         # #    "scale=224:224",
#         # #    '-c:a', 'copy',
#         # #    'fps=fps=30',  # frames per second
#         #     output_video_path_ori,
#         #     ]
#
#         ### ori compressed ---------->
#         # command = ['ffmpeg',
#         #            '-y',  # (optional) overwrite output file if it exists
#         #            '-i', input_video_path,
#         #            '-filter:v',
#         #            'scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'',  # scale to 224
#         #            '-map', '0:v',
#         #            '-r', '3',  # frames per second
#         #            output_video_path_ori,
#         #            ]
#         ########### <----------------
#
#         ############# for extract images ##############
#         command = ['ffmpeg',
#         	'-y',  # (optional) overwrite output file if it exists
#         	'-i', input_video_path,
#         	# '-vf',
#         	# 'fps=3',
#         	output_img_path,
#         ]
#         ########## end extract images ###################################
#
#         ######### for extract audio ----->
#         # ffmpeg -i /playpen-iop/yblin/v1-2/train/v_XazKuBawFCM.mp4 -map 0:a -f segment -segment_time 10 -acodec pcm_s16le -ac 1 -ar 16000 /playpen-iop/yblin/v1-2/train_audio/output_%03d.wav
#         # ffmpeg -y -i /playpen-iop/yblin/yk2/raw_videos_all/low_all_val/EpNUSTO2BI4.mp4 -map 0:a -f segment -segment_time 10000000 -acodec pcm_s16le -ac 1 -ar 16000 /playpen-iop/yblin/yk2/audio_raw_val/EpNUSTO2BI4.wav
#         # command = ['ffmpeg',
#         #            '-y',  # (optional) overwrite output file if it exists
#         #            '-i', input_video_path,
#         #            '-acodec', 'pcm_s16le', '-ac', '1',
#         #            '-ar', '16000',  # resample
#         #            output_audio2_path,
#         #            ]
#         ### <------
#
#         ######### for extract audio 2 ###########
#         # command = ['ffmpeg',
#         # 	'-y',  # (optional) overwrite output file if it exists
#         # 	'-i', input_video_path,
#         # 	'-map','0:a', '-f', 'segment',
#         # 	'-segment_time', '10000000', # seconds here
#         # 	'-acodec', 'pcm_s16le', '-ac', '1',
#         # 	'-ar', '16000', # resample
#         # 	output_audio_path,
#         # ]
#         #####################
#
#         # command= [
#         # 	'mkdir',
#         # 	output_video_path,  # (optional) overwrite output file if it exists
#         # ]
#
#         print(command)
#
#         # ffmpeg -y -i /playpen-iop/yblin/v1-2/val/v_6VT2jBflMAM.mp4 -vf  fps=3 /playpen-iop/yblin/v1-2/val_low_scale/v_6VT2jBflMAM.mp4
#         ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         out, err = ffmpeg.communicate()
#         retcode = ffmpeg.poll()
#     # print something above for debug
#     except Exception as e:
#         raise e

def compress(paras):
    input_video_path, output_video_path = paras
    try:
        command = ['ffmpeg',
                   '-y',  # (optional) overwrite output file if it exists
                   '-i', input_video_path,
                   '-filter:v',
                   'scale=\'if(gt(a,1),trunc(oh*a/2)*2,224)\':\'if(gt(a,1),224,trunc(ow*a/2)*2)\'',  # scale to 224
                   '-map', '0:v',
                   '-r', '3',  # frames per second
                   output_video_path,
                   ]
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        retcode = ffmpeg.poll()
        # print something above for debug
    except Exception as e:
        raise e

# def prepare_input_output_pairs(input_root, output_root):
#     input_video_path_list = []
#     output_video_path_list = []
#     for root, dirs, files in os.walk(input_root):
#         for file_name in files:
#             input_video_path = os.path.join(root, file_name)
#             output_video_path = os.path.join(output_root, file_name)
#             if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
#                 pass
#             else:
#                 input_video_path_list.append(input_video_path)
#                 output_video_path_list.append(output_video_path)
#     return input_video_path_list, output_video_path_list


# def prepare_input_output_pairs(input_root, output_root):
#     input_video_path_list = []
#     output_video_path_list = []
#
#     for split in ['train', 'val']:
#         data_json = f'/playpen-storage/mmiemon/ego4d/data/annotations/nlq_{split}.json'
#         with open(data_json, mode="r", encoding="utf-8") as f:
#             data = json.load(f)['videos']
#
#         for video_datum in data:
#             input_video_path_list.append(f'{input_root}/{video_datum["video_uid"]}.mp4')
#             output_video_path_list.append(f'{output_root}/{video_datum["video_uid"]}.mp4')
#
#     return input_video_path_list, output_video_path_list

def prepare_input_output_pairs(input_root, output_root):
    input_video_path_list = []
    output_video_path_list = []

    for split in ['train', 'val', 'test_unannotated']:
        json_file = f'/playpen-storage/mmiemon/ego4d/Ego4d/annotations/v1/annotations/fho_oscc-pnr_{split}.json'
        with open(json_file, 'r') as f:
            data = json.load(f)
            data = data['clips']
        for video_datum in data:
            out_path = f'{output_root}/{video_datum["video_uid"]}.mp4'
            if os.path.exists(out_path) or out_path in output_video_path_list:
                continue
            input_video_path_list.append(f'{input_root}/{video_datum["video_uid"]}.mp4')
            output_video_path_list.append(f'{output_root}/{video_datum["video_uid"]}.mp4')

    return input_video_path_list, output_video_path_list


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Compress video for speed-up')
    # parser.add_argument('--input_root', type=str, help='input root')
    # parser.add_argument('--output_root', type=str, help='output root')
    # args = parser.parse_args()
    #
    # input_root = args.input_root
    # output_root = args.output_root
    input_root = '/playpen-storage/mmiemon/ego4d/data/v1/full_scale'
    output_root = '/playpen-storage/mmiemon/ego4d/data/v1/full_scale_fps_3'
    assert input_root != output_root

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    input_video_path_list, output_video_path_list = prepare_input_output_pairs(input_root, output_root)

    print(len(input_video_path_list), len(output_video_path_list))

    print("Total video need to process: {}".format(len(input_video_path_list)))
    num_works = cpu_count()
    print("Begin with {}-core logical processor.".format(num_works))

    pool = Pool(num_works)
    data_dict_list = pool.map(compress,
                              [(input_video_path, output_video_path) for
                               input_video_path, output_video_path in
                               zip(input_video_path_list, output_video_path_list)])
    pool.close()
    pool.join()
    #
    # print("Compress finished, wait for checking files...")
    # for input_video_path, output_video_path in zip(input_video_path_list, output_video_path_list):
    #     if os.path.exists(input_video_path):
    #         if os.path.exists(output_video_path) is False or os.path.getsize(output_video_path) < 1.:
    #             shutil.copyfile(input_video_path, output_video_path)
    #             print("Copy and replace file: {}".format(output_video_path))

#ffmpeg -i /playpen-storage/mmiemon/ego4d/data/v1/clips_fps_3_224/59d86631-fec6-419f-ad65-67ad0a3c5144.mp4 /playpen-storage/mmiemon/ego4d/data/v1/clips_fps_3_224_frames/59d86631-fec6-419f-ad65-67ad0a3c5144/%04d.jpg