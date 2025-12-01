"""
Realtime Emotion Detection using trained SVM
Run with:
python realtime_detect.py --model svm_emotion_model.joblib --scaler scaler.joblib --camera 0
"""

import argparse
import cv2
import numpy as np
import joblib
import time

# Constants
TARGET_SIZE = (48, 48)
HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EMOTION_NAME = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

def preprocess_face(face_gray):
    face_resized = cv2.resize(face_gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    face_norm = face_resized.astype(np.float32) / 255.0
    return face_norm.flatten()


def run(model_path, scaler_path, camera_index=0):
    print("Loading model and scaler...")
    svm = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_index}")
        return

    print("Realtime detection started. Press 'q' to quit, 's' to save snapshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            pad = int(w * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face_roi = gray[y1:y2, x1:x2]
            flat = preprocess_face(face_roi)
            flat_scaled = scaler.transform([flat])

            prob = svm.predict_proba(flat_scaled)[0]
            label_idx = int(np.argmax(prob))
            label = EMOTION_NAME.get(label_idx, str(label_idx))
            conf = prob[label_idx] * 100

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.1f}%)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Realtime Emotion Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print("Saved:", fname)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained SVM model")
    parser.add_argument("--scaler", required=True, help="Path to scaler file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    run(args.model, args.scaler, args.camera)
