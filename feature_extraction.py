# feature_extraction.py

import os
import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm

# Load EfficientNetB0 (without final classification layer)
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features_from_frames(frame_folder, max_frames=30):
    frames = sorted(os.listdir(frame_folder))[:max_frames]
    features = []

    for frame in frames:
        img_path = os.path.join(frame_folder, frame)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        feature = model.predict(x, verbose=0)[0]
        features.append(feature)

    # Pad with zeros if less than max_frames
    while len(features) < max_frames:
        features.append(np.zeros_like(features[0]))

    return np.array(features)

def extract_and_save_all_features(frame_data_dir, save_name_prefix='features'):
    data = []
    labels = []

    for label, folder in enumerate(['normal', 'crime']):  # normal = 0, crime = 1
        class_dir = os.path.join(frame_data_dir, folder)
        print(f"Processing {folder} videos...")
        for video_folder in tqdm(os.listdir(class_dir)):
            video_path = os.path.join(class_dir, video_folder)
            if os.path.isdir(video_path):
                features = extract_features_from_frames(video_path)
                data.append(features)
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    # Save the extracted features and labels
    np.save(f'{save_name_prefix}.npy', data)
    np.save(f'{save_name_prefix}_labels.npy', labels)
    print("✅ Features and labels saved!")

# Entry point
if __name__ == "__main__":
    print("🚀 Feature extraction started...")
    extract_and_save_all_features('frame_data')
