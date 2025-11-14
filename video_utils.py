import cv2
import os

def extract_frames_from_video(video_path, output_folder, frames_per_second=10):
    """
    Extracts `frames_per_second` frames from a video and saves them to output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"Cannot access video")
        return

    frame_interval = int(fps / frames_per_second)
    count = 0
    frame_count = 0

    success, frame = cap.read()
    while success:
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        success, frame = cap.read()
        count += 1

    cap.release()
    print(f"✅ Extracted {frame_count} frames at {frames_per_second} fps from {video_path}")
