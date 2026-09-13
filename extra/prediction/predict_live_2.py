import sys
import os

from collection_script import draw_landmarks
from utils.drawing_utils import draw_confirmation_overlay, draw_info_line, draw_mode_banner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import time
import joblib
import numpy as np
import mediapipe as mp

from preprocessing.feature_utils import normalize_landmarks
from controls.gesture_actions import perform_action, get_status

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_PATH = "models/gesture_model_final_1.pkl"
HAND_LANDMARKER_PATH = "models/hand_landmarker.task"
MAX_NUM_HANDS = 1

PREDICTION_STABLE_FRAMES = 5
CONFIDENCE_THRESHOLD = 0.80
ACTION_COOLDOWN = 1.5

WINDOW_NAME = "Live Gesture Prediction"

# --------------------------------------------------
# MediaPipe Tasks aliases
# --------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def get_prediction_and_confidence(model, features_array):
    probs = model.predict_proba(features_array)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    predicted_label = model.classes_[pred_idx]
    return predicted_label, confidence


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Trained ML model not found: {MODEL_PATH}")
        return

    if not os.path.exists(HAND_LANDMARKER_PATH):
        print(f"Hand landmarker model not found: {HAND_LANDMARKER_PATH}")
        return

    model = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    last_prediction = None
    stable_count = 0
    stable_prediction = "No stable prediction"
    current_confidence = 0.0

    last_action_time = 0.0
    last_triggered_label = None

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

            current_prediction = "No Hand Detected"
            current_confidence = 0.0

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, hand_landmarks)

                features = normalize_landmarks(hand_landmarks)
                features_array = np.array(features, dtype=np.float32).reshape(1, -1)

                predicted_label, confidence = get_prediction_and_confidence(model, features_array)
                current_confidence = confidence

                if predicted_label == "unknown" or confidence < CONFIDENCE_THRESHOLD:
                    current_prediction = "unknown"
                else:
                    current_prediction = predicted_label

                if current_prediction not in ["unknown", "No Hand Detected"]:
                    if current_prediction == last_prediction:
                        stable_count += 1
                    else:
                        last_prediction = current_prediction
                        stable_count = 1

                    if stable_count >= PREDICTION_STABLE_FRAMES:
                        stable_prediction = current_prediction
                else:
                    last_prediction = None
                    stable_count = 0
                    stable_prediction = "No stable prediction"
                    last_triggered_label = None

            else:
                last_prediction = None
                stable_count = 0
                stable_prediction = "No stable prediction"
                last_triggered_label = None

            current_time = time.time()
            state = get_status()

            if stable_prediction not in ["No stable prediction", "unknown"]:
                in_confirmation = state["mode"] == "CONFIRMATION"

                can_trigger = False

                if in_confirmation:
                    # Allow confirm/cancel immediately
                    can_trigger = stable_prediction in ["thumbs_up", "thumbs_down"]
                else:
                    can_trigger = (
                        current_time - last_action_time > ACTION_COOLDOWN
                        and stable_prediction != last_triggered_label
                    )

                if can_trigger:
                    result_info = perform_action(stable_prediction)

                    # If entering confirmation mode, do not block immediate thumbs_up/down
                    if result_info["mode"] == "CONFIRMATION":
                        last_action_time = 0.0
                        last_triggered_label = None
                    else:
                        last_action_time = current_time
                        last_triggered_label = stable_prediction

            state = get_status()
            shown_title = state["title"] if state["title"] else "No active window title"
            shown_title = shown_title[:65]
            pending_action_text = state["pending_action"] if state["pending_action"] else "None"

            draw_info_line(frame, f"Current: {current_prediction}", 35, (0, 255, 255), 0.8, 2)
            draw_info_line(frame, f"Stable: {stable_prediction}", 70, (0, 255, 0), 0.8, 2)
            draw_info_line(frame, f"Stable Frames: {stable_count}/{PREDICTION_STABLE_FRAMES}", 105, (255, 255, 0), 0.7, 2)
            draw_info_line(frame, f"Confidence: {current_confidence:.2f}", 140, (255, 255, 255), 0.7, 2)
            draw_info_line(frame, f"Threshold: {CONFIDENCE_THRESHOLD:.2f}", 175, (180, 180, 180), 0.7, 2)

            draw_info_line(frame, f"Mode: {state['mode']}", 210, (0, 200, 255), 0.7, 2)
            draw_info_line(frame, f"Base Mode: {state['base_mode']}", 245, (200, 220, 0), 0.7, 2)
            draw_info_line(frame, f"Pending Action: {pending_action_text}", 280, (255, 180, 255), 0.65, 2)
            draw_info_line(frame, f"Window: {shown_title}", 315, (200, 200, 200), 0.6, 2)
            draw_info_line(frame, f"Status: {state['status_message']}", 350, (150, 255, 150), 0.6, 2)
            draw_info_line(frame, "Press 'q' to quit", 385, (200, 200, 200), 0.7, 2)

            draw_mode_banner(frame, state)
            draw_confirmation_overlay(frame, state)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()