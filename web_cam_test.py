
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load the trained model
model = load_model('crime_detection_model_gru.keras')

# Load EfficientNetB0 for feature extraction
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Extract features from a single frame
def extract_features_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    feature = feature_extractor.predict(x, verbose=0)[0]
    return feature

# Live webcam crime detection
def detect_crime_webcam(save_output=False):
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    # Optional: Save output video
    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        out = cv2.VideoWriter('output_video/live_output.mp4', fourcc, fps, (width, height))

    print("📸 Starting webcam... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Unable to access webcam.")
            break

        feature = extract_features_from_frame(frame)
        feature = np.expand_dims([feature], axis=0)
        prediction = model.predict(feature, verbose=0)[0][0]

        # Display prediction and bounding box
        if prediction > 0.5:
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 255), 4)
            cv2.putText(frame, f"CRIME ({prediction:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, f"Normal ({prediction:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        if save_output and out:
            out.write(frame)

        cv2.imshow("🔴 Live Crime Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(" Webcam stopped.")

# Run
if __name__ == "__main__":
    detect_crime_webcam(save_output=True)  # Set to False if you don't want to save the video
