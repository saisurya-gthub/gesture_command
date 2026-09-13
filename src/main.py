import cv2
import time

from controls.gesture_actions import perform_action, get_status
from controls.rule_controller import handle_rule_based, reset_rule_state
from ml.live_predictor import LiveGesturePredictor
from utils.drawing_utils import (
    draw_landmarks,
    draw_info_line,
    draw_mode_banner,
    draw_confirmation_overlay,
)

WINDOW_NAME = "Gesture Based System"
ACTION_COOLDOWN = 0.6

# Camera / display size
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Static ML gestures
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

# Freeze tuning
STATIC_TRIGGER_CONFIDENCE = 0.60
STATIC_TRIGGER_STABLE_COUNT = 2
RULE_FREEZE_FRAMES = 5


def is_static_candidate(current_prediction, confidence, stable_count):
    """Check whether a valid ML static gesture is forming."""
    return (
        current_prediction in KNOWN_STATIC_GESTURES
        and confidence >= STATIC_TRIGGER_CONFIDENCE
        and stable_count >= STATIC_TRIGGER_STABLE_COUNT
    )


def main():
    """Run integrated rule-based + ML-based gesture system."""
    predictor = LiveGesturePredictor()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    # Set camera capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Create a resizable larger display window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    last_action_time = 0.0
    prev_time = time.time()
    rule_freeze_frames = 0

    # Release-to-rearm lock for static gestures
    armed_static_gesture = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)

        # Keep frame at the intended display size
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        prediction = predictor.process_frame(frame)
        hand_landmarks = prediction["hand_landmarks"]
        current_prediction = prediction.get("current_prediction", "unknown")
        stable_prediction = prediction.get("stable_prediction", "unknown")
        confidence = prediction.get("confidence", 0.0)
        stable_count = prediction.get("stable_count", 0)

        # Re-arm when stable gesture is gone
        if stable_prediction == "unknown":
            armed_static_gesture = None

        if hand_landmarks is not None:
            draw_landmarks(frame, hand_landmarks)
        else:
            reset_rule_state()
            rule_freeze_frames = 0
            armed_static_gesture = None

        state = get_status()

        rule_active = False
        rule_name = "None"
        current_time = time.time()

        # Freeze rule logic for a few frames when static gesture is forming
        static_candidate_active = is_static_candidate(
            current_prediction=current_prediction,
            confidence=confidence,
            stable_count=stable_count,
        )

        if hand_landmarks is not None and static_candidate_active:
            rule_freeze_frames = RULE_FREEZE_FRAMES
        elif rule_freeze_frames > 0:
            rule_freeze_frames -= 1

        rules_frozen = rule_freeze_frames > 0

        # --------------------------------------------------
        # 1. Confirmation mode has highest priority
        # --------------------------------------------------
        if state["mode"] == "CONFIRMATION":
            rule_active = False
            rule_name = "Frozen (Confirmation)"

            if stable_prediction in ["thumbs_up", "thumbs_down"]:
                # Release-to-rearm also applied here
                if armed_static_gesture != stable_prediction:
                    result = perform_action(stable_prediction)
                    if result["handled"]:
                        last_action_time = 0.0
                        rule_freeze_frames = 0
                        armed_static_gesture = stable_prediction

        # --------------------------------------------------
        # 2. Normal mode
        # --------------------------------------------------
        else:
            if hand_landmarks is not None:
                rule_active, rule_name = handle_rule_based(
                    hand_landmarks,
                    rules_enabled=not rules_frozen
                )
            else:
                rule_active = False
                rule_name = "No Hand"

            # Static ML gestures with release-to-rearm
            if stable_prediction in KNOWN_STATIC_GESTURES:
                can_trigger = (
                    armed_static_gesture != stable_prediction
                    and (current_time - last_action_time > ACTION_COOLDOWN)
                )

                if can_trigger:
                    result = perform_action(stable_prediction)

                    if result["mode"] == "CONFIRMATION":
                        last_action_time = 0.0
                        rule_freeze_frames = 0
                        armed_static_gesture = stable_prediction
                    else:
                        last_action_time = current_time
                        rule_freeze_frames = 0
                        armed_static_gesture = stable_prediction

        state = get_status()
        pending_action_text = state["pending_action"] if state["pending_action"] else "None"
        shown_title = state["title"] if state["title"] else "No active window title"
        shown_title = shown_title[:65]

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        draw_info_line(frame, f"FPS: {fps:.1f}", 35, (0, 255, 255), 0.8, 2)
        draw_info_line(frame, f"Current: {current_prediction}", 70, (0, 255, 255), 0.8, 2)
        draw_info_line(frame, f"Stable: {stable_prediction}", 105, (0, 255, 0), 0.8, 2)
        draw_info_line(frame, f"Stable Frames: {stable_count}/5", 140, (255, 255, 0), 0.7, 2)
        draw_info_line(frame, f"Confidence: {confidence:.2f}", 175, (255, 255, 255), 0.7, 2)

        draw_info_line(frame, f"Mode: {state['mode']}", 210, (0, 200, 255), 0.7, 2)
        draw_info_line(frame, f"Base Mode: {state['base_mode']}", 245, (200, 220, 0), 0.7, 2)
        draw_info_line(frame, f"Pending Action: {pending_action_text}", 280, (255, 180, 255), 0.65, 2)
        draw_info_line(frame, f"Rule Active: {rule_active}", 315, (0, 255, 150), 0.65, 2)
        draw_info_line(frame, f"Rule Name: {rule_name}", 350, (0, 255, 150), 0.65, 2)
        draw_info_line(frame, f"Rules Frozen: {rules_frozen}", 385, (255, 200, 0), 0.65, 2)
        draw_info_line(frame, f"Static Armed: {armed_static_gesture}", 420, (255, 220, 120), 0.65, 2)
        draw_info_line(frame, f"Window: {shown_title}", 455, (200, 200, 200), 0.6, 2)
        draw_info_line(frame, f"Status: {state['status_message']}", 490, (150, 255, 150), 0.6, 2)
        draw_info_line(frame, "Press 'q' to quit", 525, (200, 200, 200), 0.7, 2)

        draw_mode_banner(frame, state)
        draw_confirmation_overlay(frame, state)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    predictor.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()