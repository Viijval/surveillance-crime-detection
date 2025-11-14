import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from collections import deque

# Load GRU model
model = load_model('crime_detection_model_gru.keras')

# Load EfficientNetB0 as feature extractor
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Extract CNN feature from one frame
def extract_features_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    feature = feature_extractor.predict(x, verbose=0)[0]
    return feature

# Prediction function
def predict_video(video_path, output_path='output_video/output.mp4'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Failed to open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_buffer = deque(maxlen=30)
    raw_frames = []
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frames.append(frame.copy())
        feature = extract_features_from_frame(frame)
        frame_buffer.append(feature)

        # Only start predicting after 30 frames are available
        if len(frame_buffer) == 30:
            input_sequence = np.expand_dims(np.array(frame_buffer), axis=0)
            prediction = model.predict(input_sequence, verbose=0)[0][0]
            predictions.append(prediction)

            # Draw on the 30th frame
            output_frame = frame.copy()
            color = (0, 255, 0)
            label = f"Normal ({prediction:.2f})"

            if prediction > 0.5:
                color = (0, 0, 255)
                label = f"CRIME ({prediction:.2f})"

            cv2.rectangle(output_frame, (10, 10), (frame_width - 10, frame_height - 10), color, 4)
            cv2.putText(output_frame, label, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            out.write(output_frame)

    cap.release()
    out.release()

    if not predictions:
        print("⚠ No sequences processed. Cannot compute predictions.")
        return

    print(f" Processed video saved at: {output_path}")
    print(f" Max confidence: {np.max(predictions):.4f}")

   # Plot confidence graph
    plt.figure(figsize=(12, 5))
    plt.plot(predictions, label='Prediction Confidence')
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.title('Video Crime Prediction')
    plt.xlabel('Prediction Count')
    plt.ylabel('Confidence (0-1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output_video/confidence_plot.png')
    plt.show()

# Run
if __name__ == "__main__":
    video_path = 'test_videos/crime_7.mp4'
    predict_video(video_path)

