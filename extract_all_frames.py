# extract_all_frames.py

import os
from utils.video_utils import extract_frames_from_video

video_root = 'dataset/'
output_root = 'frame_data/'

categories = ['crime', 'normal']

for category in categories:
    input_folder = os.path.join(video_root, category)
    output_folder = os.path.join(output_root, category)

    for video_name in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_name)
        video_output = os.path.join(output_folder, video_name.split('.')[0])
        extract_frames_from_video(video_path, video_output)
