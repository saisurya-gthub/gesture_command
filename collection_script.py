import os
import csv
import time
import cv2
import mediapipe as mp

from src.preprocessing.feature_utils import normalize_landmarks
from src.utils.drawing_utils import draw_landmarks

# File paths
DATASET_FILE = "data/gestures_dataset.csv"
MODEL_PATH = "models/hand_landmarker.task"
MAX_NUM_HANDS = 1

# MediaPipe aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def create_csv_if_not_exists(filename: str) -> None:
    """Create dataset CSV with header if missing."""
    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)

    if not os.path.exists(filename):
        header = ["label"]
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])

        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def save_landmarks_to_csv(label: str, landmarks, filename: str) -> None:
    """Save one normalized hand sample."""
    normalized_features = normalize_landmarks(landmarks)
    row = [label] + normalized_features

    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    """Collect gesture samples and save them to CSV."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        print("Place hand_landmarker.task inside the models folder.")
        return

    create_csv_if_not_exists(DATASET_FILE)

    label = input("Enter gesture label: ").strip()
    if not label:
        print("Label cannot be empty.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    sample_count = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            cv2.putText(
                frame,
                f"Label: {label} | Saved: {sample_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press 's' to save, 'q' to quit",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            hand_found = False
            first_hand_landmarks = None

            if result.hand_landmarks:
                first_hand_landmarks = result.hand_landmarks[0]
                hand_found = True

                for hand_landmarks in result.hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

            cv2.imshow("Dataset Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if hand_found and first_hand_landmarks is not None:
                    save_landmarks_to_csv(label, first_hand_landmarks, DATASET_FILE)
                    sample_count += 1
                    print(f"Saved sample {sample_count} for label '{label}'")
                else:
                    print("No hand detected. Sample not saved.")

            elif key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()