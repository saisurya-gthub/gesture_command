import cv2
import time
import math
import pyautogui
import mediapipe as mp
from mediapipe.tasks.python import vision

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = "models/hand_landmarker.task"

CAM_W = 1280
CAM_H = 720

FRAME_MARGIN_X = 150
FRAME_MARGIN_Y = 100

SMOOTHING = 0.80
SPEED_FACTOR = 0.40

MAX_MISSING_FRAMES = 5

# Three-finger activation thresholds
ON_THRESHOLD_48 = 0.08
ON_THRESHOLD_812 = 0.08
ON_THRESHOLD_412 = 0.14

OFF_THRESHOLD_48 = 0.11
OFF_THRESHOLD_812 = 0.11
OFF_THRESHOLD_412 = 0.16

TOUCH_STABLE_FRAMES = 2
RELEASE_STABLE_FRAMES = 2

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

SCREEN_W, SCREEN_H = pyautogui.size()

# -----------------------------
# GLOBAL STATE
# -----------------------------
prev_x, prev_y = 0, 0
cursor_active = False
touch_count = 0
release_count = 0
last_timestamp_ms = 0

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


def create_hand_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return HandLandmarker.create_from_options(options)


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def distance_2d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def map_range(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    value = max(in_min, min(value, in_max))
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def get_monotonic_timestamp_ms():
    global last_timestamp_ms
    now = int(time.time() * 1000)
    if now <= last_timestamp_ms:
        now = last_timestamp_ms + 1
    last_timestamp_ms = now
    return now


def update_three_finger_state(thumb_tip, index_tip, middle_tip):
    global cursor_active, touch_count, release_count

    d1 = distance_2d(thumb_tip, index_tip)      # 4-8
    d2 = distance_2d(index_tip, middle_tip)     # 8-12
    d3 = distance_2d(thumb_tip, middle_tip)     # 4-12

    touching = (
        d1 < ON_THRESHOLD_48 and
        d2 < ON_THRESHOLD_812 and
        d3 < ON_THRESHOLD_412
    )

    separated = (
        d1 > OFF_THRESHOLD_48 or
        d2 > OFF_THRESHOLD_812 or
        d3 > OFF_THRESHOLD_412
    )

    if touching:
        touch_count += 1
        release_count = 0
    elif separated:
        release_count += 1
        touch_count = 0
    else:
        touch_count = 0
        release_count = 0

    if touch_count >= TOUCH_STABLE_FRAMES:
        cursor_active = True

    if release_count >= RELEASE_STABLE_FRAMES:
        cursor_active = False

    return cursor_active, (d1, d2, d3)


def move_cursor(index_tip):
    global prev_x, prev_y

    cam_x = int(index_tip.x * CAM_W)
    cam_y = int(index_tip.y * CAM_H)

    cam_x = max(FRAME_MARGIN_X, min(cam_x, CAM_W - FRAME_MARGIN_X))
    cam_y = max(FRAME_MARGIN_Y, min(cam_y, CAM_H - FRAME_MARGIN_Y))

    screen_x = map_range(cam_x, FRAME_MARGIN_X, CAM_W - FRAME_MARGIN_X, 0, SCREEN_W)
    screen_y = map_range(cam_y, FRAME_MARGIN_Y, CAM_H - FRAME_MARGIN_Y, 0, SCREEN_H)

    target_x = prev_x + (screen_x - prev_x) * SPEED_FACTOR
    target_y = prev_y + (screen_y - prev_y) * SPEED_FACTOR

    curr_x = int(prev_x * SMOOTHING + target_x * (1 - SMOOTHING))
    curr_y = int(prev_y * SMOOTHING + target_y * (1 - SMOOTHING))

    pyautogui.moveTo(curr_x, curr_y, duration=0)

    prev_x, prev_y = curr_x, curr_y
    return cam_x, cam_y, curr_x, curr_y


# -----------------------------
# MAIN
# -----------------------------
def main():
    global prev_x, prev_y

    missing_hand_frames = 0
    last_valid_landmarks = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("Error: webcam not opened")
        return

    prev_x, prev_y = SCREEN_W // 2, SCREEN_H // 2

    with create_hand_landmarker() as landmarker:
        prev_time = time.time()
        fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: frame not read")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = get_monotonic_timestamp_ms()
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            move_text = "Cursor Move: OFF"
            status_text = "No Hand"
            d1, d2, d3 = 0.0, 0.0, 0.0

            # Draw active region
            cv2.rectangle(
                frame,
                (FRAME_MARGIN_X, FRAME_MARGIN_Y),
                (CAM_W - FRAME_MARGIN_X, CAM_H - FRAME_MARGIN_Y),
                (255, 255, 0),
                2
            )

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand_landmarks = result.hand_landmarks[0]
                last_valid_landmarks = hand_landmarks
                missing_hand_frames = 0
                hand_available = True
                status_text = "Hand Detected"
            else:
                missing_hand_frames += 1
                if last_valid_landmarks is not None and missing_hand_frames <= MAX_MISSING_FRAMES:
                    hand_landmarks = last_valid_landmarks
                    hand_available = True
                    status_text = "Tracking Memory"
                else:
                    hand_available = False
                    status_text = "Tracking Lost"

            if hand_available:
                for lm in hand_landmarks:
                    x = int(lm.x * CAM_W)
                    y = int(lm.y * CAM_H)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                middle_tip = hand_landmarks[12]

                thumb_px = int(thumb_tip.x * CAM_W)
                thumb_py = int(thumb_tip.y * CAM_H)
                index_px = int(index_tip.x * CAM_W)
                index_py = int(index_tip.y * CAM_H)
                middle_px = int(middle_tip.x * CAM_W)
                middle_py = int(middle_tip.y * CAM_H)

                cv2.line(frame, (thumb_px, thumb_py), (index_px, index_py), (255, 0, 255), 2)
                cv2.line(frame, (index_px, index_py), (middle_px, middle_py), (255, 0, 255), 2)
                cv2.line(frame, (thumb_px, thumb_py), (middle_px, middle_py), (255, 0, 255), 2)

                can_move, distances = update_three_finger_state(
                    thumb_tip, index_tip, middle_tip
                )
                d1, d2, d3 = distances

                if can_move:
                    cam_x, cam_y, _, _ = move_cursor(index_tip)
                    cv2.circle(frame, (cam_x, cam_y), 8, (0, 0, 255), -1)
                    move_text = "Cursor Move: ON"
                else:
                    move_text = "Cursor Move: OFF"

            # FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # Display
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, status_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, "Press Q to quit", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            cv2.putText(frame, move_text, (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"4-8: {d1:.4f}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"8-12: {d2:.4f}", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"4-12: {d3:.4f}", (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Gesture Cursor Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()