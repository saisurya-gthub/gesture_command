import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import time
import joblib
import numpy as np
import mediapipe as mp

from preprocessing.feature_utils import normalize_landmarks
from controls.gestures_action import perform_action, get_status

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_PATH = "models/gesture_model_final_1.pkl"
HAND_LANDMARKER_PATH = "models/hand_landmarker.task"
MAX_NUM_HANDS = 1

PREDICTION_STABLE_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.80
ACTION_COOLDOWN = 0.5  # seconds

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


# --------------------------------------------------
# Drawing helpers
# --------------------------------------------------
def draw_landmarks(frame, landmarks):
    """
    Draws landmark points and hand connections on frame.
    """
    h, w, _ = frame.shape
    points = []

    for lm in landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))
        cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

    for start_idx, end_idx in HAND_CONNECTIONS:
        x1, y1 = points[start_idx]
        x2, y2 = points[end_idx]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


def draw_info_line(frame, text, y, color=(255, 255, 255), scale=0.65, thickness=2):
    """
    Draws one line of overlay text.
    """
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


def draw_mode_banner(frame, state):
    """
    Draws a small banner on top-right showing the current mode.
    """
    mode = state.get("mode", "UNKNOWN")
    pending_action = state.get("pending_action", None)

    if mode == "CONFIRMATION":
        banner_text = f"MODE: {mode}"
        banner_color = (0, 0, 255)
    elif mode == "BROWSER":
        banner_text = f"MODE: {mode}"
        banner_color = (255, 180, 0)
    elif mode == "PRESENTATION":
        banner_text = f"MODE: {mode}"
        banner_color = (255, 0, 255)
    else:
        banner_text = f"MODE: {mode}"
        banner_color = (0, 180, 255)

    h, w, _ = frame.shape
    box_w = 280
    box_h = 45
    x1 = w - box_w - 10
    y1 = 10
    x2 = w - 10
    y2 = y1 + box_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), banner_color, -1)
    cv2.putText(
        frame,
        banner_text,
        (x1 + 10, y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if pending_action:
        extra_text = f"PENDING: {pending_action}"
        cv2.rectangle(frame, (x1, y2 + 8), (x2, y2 + 43), (50, 50, 50), -1)
        cv2.putText(
            frame,
            extra_text,
            (x1 + 10, y2 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


def draw_confirmation_overlay(frame, state):
    """
    Draws a larger overlay when confirmation mode is active.
    """
    if state.get("mode") != "CONFIRMATION":
        return

    h, w, _ = frame.shape

    overlay_x1 = 40
    overlay_y1 = h - 155
    overlay_x2 = w - 40
    overlay_y2 = h - 30

    cv2.rectangle(frame, (overlay_x1, overlay_y1), (overlay_x2, overlay_y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (overlay_x1, overlay_y1), (overlay_x2, overlay_y2), (0, 0, 255), 2)

    pending_action = state.get("pending_action") or "ACTION"

    cv2.putText(
        frame,
        f"{pending_action} selected - Confirmation Mode",
        (overlay_x1 + 15, overlay_y1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "thumbs_up  -> Proceed",
        (overlay_x1 + 15, overlay_y1 + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "thumbs_down -> Cancel",
        (overlay_x1 + 15, overlay_y1 + 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 120, 255),
        2
    )


# --------------------------------------------------
# Prediction helper
# --------------------------------------------------
def get_prediction_and_confidence(model, features_array):
    """
    Returns predicted class label and confidence score.
    """
    probs = model.predict_proba(features_array)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    predicted_label = model.classes_[pred_idx]
    return predicted_label, confidence


# --------------------------------------------------
# Main loop
# --------------------------------------------------
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

            if stable_prediction not in ["No stable prediction", "unknown"]:
                if (
                    current_time - last_action_time > ACTION_COOLDOWN
                    and stable_prediction != last_triggered_label
                ):
                    perform_action(stable_prediction)
                    last_action_time = current_time
                    last_triggered_label = stable_prediction

            state = get_status()
            shown_title = state["title"] if state["title"] else "No active window title"
            shown_title = shown_title[:65]

            pending_action_text = state["pending_action"] if state["pending_action"] else "None"

            draw_info_line(frame, f"Current: {current_prediction}", 35, (0, 255, 255), 0.8, 2)
            draw_info_line(frame, f"Stable: {stable_prediction}", 70, (0, 255, 0), 0.8, 2)
            draw_info_line(
                frame,
                f"Stable Frames: {stable_count}/{PREDICTION_STABLE_FRAMES}",
                105,
                (255, 255, 0),
                0.7,
                2
            )
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