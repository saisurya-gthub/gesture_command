import cv2
import time
import math
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# SETTINGS
# -----------------------------
MODEL_PATH = "models/hand_landmarker.task"

SMOOTHING = 0.8
PINCH_THRESHOLD = 0.05
CLICK_COOLDOWN = 0.8

# reduce cursor jump by using only part of camera frame
CAM_W = 1280
CAM_H = 720

FRAME_MARGIN_X = 150
FRAME_MARGIN_Y = 100

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

SCREEN_W, SCREEN_H = pyautogui.size()

# previous cursor position for smoothing
prev_x, prev_y = 0, 0

# click cooldown
last_click_time = 0

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
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
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


def move_cursor(index_tip):
    global prev_x, prev_y

    cam_x = int(index_tip.x * CAM_W)
    cam_y = int(index_tip.y * CAM_H)

    cam_x = max(FRAME_MARGIN_X, min(cam_x, CAM_W - FRAME_MARGIN_X))
    cam_y = max(FRAME_MARGIN_Y, min(cam_y, CAM_H - FRAME_MARGIN_Y))

    screen_x = map_range(cam_x, FRAME_MARGIN_X, CAM_W - FRAME_MARGIN_X, 0, SCREEN_W)
    screen_y = map_range(cam_y, FRAME_MARGIN_Y, CAM_H - FRAME_MARGIN_Y, 0, SCREEN_H)

    SPEED_FACTOR = 0.4

    target_x = prev_x + (screen_x - prev_x) * SPEED_FACTOR
    target_y = prev_y + (screen_y - prev_y) * SPEED_FACTOR

    curr_x = int(prev_x * SMOOTHING + target_x * (1 - SMOOTHING))
    curr_y = int(prev_y * SMOOTHING + target_y * (1 - SMOOTHING))

    pyautogui.moveTo(curr_x, curr_y, duration=0)

    prev_x, prev_y = curr_x, curr_y
    return cam_x, cam_y, curr_x, curr_y


def detect_left_click(thumb_tip, index_tip):
    global last_click_time

    pinch_dist = distance_2d(thumb_tip, index_tip)
    now = time.time()

    if pinch_dist < PINCH_THRESHOLD and (now - last_click_time) > CLICK_COOLDOWN:
        pyautogui.click()
        last_click_time = now
        return True, pinch_dist

    return False, pinch_dist


# -----------------------------
# MAIN
# -----------------------------
def main():
    global prev_x, prev_y

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    if not cap.isOpened():
        print("Error: webcam not opened")
        return

    prev_x, prev_y = SCREEN_W // 2, SCREEN_H // 2

    with create_hand_landmarker() as landmarker:
        frame_index = 0
        prev_time = time.time()
        fps = 30.0

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

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            click_text = "No Click"
            pinch_dist_value = 0.0

            # draw active area
            cv2.rectangle(
                frame,
                (FRAME_MARGIN_X, FRAME_MARGIN_Y),
                (CAM_W - FRAME_MARGIN_X, CAM_H - FRAME_MARGIN_Y),
                (255, 255, 0),
                2
            )

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand_landmarks = result.hand_landmarks[0]

                # draw all landmarks
                for lm in hand_landmarks:
                    x = int(lm.x * CAM_W)
                    y = int(lm.y * CAM_H)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # important landmarks
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]

                # move cursor with index fingertip
                cam_x, cam_y, scr_x, scr_y = move_cursor(index_tip)

                cv2.circle(frame, (cam_x, cam_y), 8, (0, 0, 255), -1)

                # detect pinch click
                clicked, pinch_dist_value = detect_left_click(thumb_tip, index_tip)
                if clicked:
                    click_text = "Left Click"

                # draw line between thumb and index
                thumb_px = int(thumb_tip.x * CAM_W)
                thumb_py = int(thumb_tip.y * CAM_H)
                index_px = int(index_tip.x * CAM_W)
                index_py = int(index_tip.y * CAM_H)
                cv2.line(frame, (thumb_px, thumb_py), (index_px, index_py), (255, 0, 255), 2)

            # calculate fps
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # show info
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"Click: {click_text}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Pinch Dist: {pinch_dist_value:.4f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, "Press Q to quit", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            cv2.imshow("Gesture Cursor Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()