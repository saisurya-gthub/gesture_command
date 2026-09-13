import os
import time
from collections import deque, Counter

import joblib
import numpy as np
import mediapipe as mp

from preprocessing.feature_utils import normalize_landmarks

MODEL_PATH = "models/gesture_model_final_1.pkl"
HAND_LANDMARKER_PATH = "models/hand_landmarker.task"
MAX_NUM_HANDS = 1

# General prediction threshold
CONFIDENCE_THRESHOLD = 0.65

# Hold frames for intentional static gestures
STATIC_GESTURE_HOLD_FRAMES = 5

# Recent prediction smoothing
PREDICTION_HISTORY_SIZE = 5
MIN_HISTORY_FOR_VOTE = 3

# Known ML static gestures
KNOWN_STATIC_GESTURES = {
    "thumbs_up",
    "thumbs_down",
    "v_sign",
    "wrapped_thumb",
    "fist",
    "next",
    "previous",
    "ok_sign",
    "open_palm",
}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class LiveGesturePredictor:
    """Handle ML gesture prediction with hold-based stability and short-term smoothing."""

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained ML model not found: {MODEL_PATH}")

        if not os.path.exists(HAND_LANDMARKER_PATH):
            raise FileNotFoundError(f"Hand landmarker model not found: {HAND_LANDMARKER_PATH}")

        self.model = joblib.load(MODEL_PATH)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=MAX_NUM_HANDS,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.last_prediction = None
        self.stable_count = 0
        self.stable_prediction = "unknown"

        self.prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)

    def get_prediction_and_confidence(self, features_array):
        """Return label and confidence from model."""
        probs = self.model.predict_proba(features_array)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_label = self.model.classes_[pred_idx]
        return predicted_label, confidence

    def get_smoothed_prediction(self):
        """Return majority-voted label from recent predictions."""
        if len(self.prediction_history) < MIN_HISTORY_FOR_VOTE:
            return None

        labels = [label for label in self.prediction_history if label != "unknown"]
        if not labels:
            return "unknown"

        label_counts = Counter(labels)
        voted_label = label_counts.most_common(1)[0][0]
        return voted_label

    def reset_prediction_state(self):
        """Reset prediction stability state."""
        self.last_prediction = None
        self.stable_count = 0
        self.stable_prediction = "unknown"
        self.prediction_history.clear()

    def process_frame(self, frame):
        """Process one frame and return ML prediction info."""
        rgb_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame[:, :, ::-1]
        )

        timestamp_ms = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(rgb_frame, timestamp_ms)

        current_prediction = "unknown"
        confidence = 0.0
        hand_landmarks = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            features = normalize_landmarks(hand_landmarks)
            features_array = np.array(features, dtype=np.float32).reshape(1, -1)

            predicted_label, confidence = self.get_prediction_and_confidence(features_array)

            if predicted_label == "unknown" or confidence < CONFIDENCE_THRESHOLD:
                current_prediction = "unknown"
            else:
                current_prediction = predicted_label

            # Keep short recent history for smoothing
            self.prediction_history.append(current_prediction)
            smoothed_prediction = self.get_smoothed_prediction()

            if smoothed_prediction in KNOWN_STATIC_GESTURES:
                if smoothed_prediction == self.last_prediction:
                    self.stable_count += 1
                else:
                    self.last_prediction = smoothed_prediction
                    self.stable_count = 1

                if self.stable_count >= STATIC_GESTURE_HOLD_FRAMES:
                    self.stable_prediction = smoothed_prediction
                else:
                    self.stable_prediction = "unknown"
            else:
                self.last_prediction = None
                self.stable_count = 0
                self.stable_prediction = "unknown"

        else:
            self.reset_prediction_state()

        return {
            "hand_landmarks": hand_landmarks,
            "current_prediction": current_prediction,
            "stable_prediction": self.stable_prediction,
            "confidence": confidence,
            "stable_count": self.stable_count,
        }

    def close(self):
        """Release landmarker resources."""
        if self.landmarker:
            self.landmarker.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass