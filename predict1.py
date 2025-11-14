import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics import precision_score, recall_score, f1_score

model = load_model('crime_detection_model_gru.keras')

base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    feature = feature_extractor.predict(x, verbose=0)[0]
    return feature

def predict_video_with_metrics(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features_buffer = []
    y_true, y_pred, confidences, prediction_times = [], [], [], []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract and store feature
        feature = extract_features_from_frame(frame)
        features_buffer.append(feature)

        # For dummy label (optional, still assumes half video = normal/crime)
        label = 0 if i < frame_count // 2 else 1

        # Time the frame inference (for graphing)
        start = time.time()
        _ = model.predict(np.expand_dims([feature], axis=0), verbose=0)  # Dummy single-frame inference
        end = time.time()
        prediction_times.append(end - start)

        # When we have 30 frames, predict the sequence
        if len(features_buffer) == 30:
            sequence_input = np.expand_dims(features_buffer, axis=0)  # shape: (1, 30, 1280)
            prediction = model.predict(sequence_input, verbose=0)[0][0]

            y_true.append(label)
            y_pred.append(1 if prediction > 0.5 else 0)
            confidences.append(prediction)

            features_buffer = []  # reset for next sequence

    cap.release()

    if not y_pred:
        print(" Not enough frames to form a sequence.")
        return

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    avg_time = np.mean(prediction_times)
    max_conf = np.max(confidences)

    print(f" Processed {len(y_pred)} sequences (each = 30 frames)")
    print(f" Precision: {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" F1 Score: {f1:.4f}")
    print(f" Avg Inference Time (per frame): {avg_time:.4f} sec")
    print(f" Max Confidence: {max_conf:.4f}")

    os.makedirs("output_video", exist_ok=True)

    # Confidence Score over Time (30-frame sequences)
    plt.figure(figsize=(12, 4))
    plt.plot(confidences, label='Sequence Confidence', color='blue')
    plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    plt.title("Confidence Score per 30-Frame Window")
    plt.xlabel("Window")
    plt.ylabel("Confidence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_video/confidence_plot.png")
    plt.show()

    # Bar chart: Precision, Recall, F1
    plt.figure(figsize=(6, 4))
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1)
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    for i, val in enumerate(values):
        plt.text(i, val + 0.02, f"{val:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("output_video/metrics_bar.png")
    plt.show()

    # Inference Time per Frame
    plt.figure(figsize=(10, 4))
    plt.plot(prediction_times, color='purple', marker='o', linewidth=1)
    plt.title("Inference Time per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_video/inference_time.png")
    plt.show()

if __name__ == "__main__":
    predict_video_with_metrics("test_videos/crime_1.mp4")
