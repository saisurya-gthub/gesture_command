import cv2
import math
import time
import pyautogui
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- SETTINGS ----------------
CAM_W = 1280
CAM_H = 720

FRAME_MARGIN_X = 150
FRAME_MARGIN_Y = 100
FRAME_REDUCTION = 100
SMOOTHING = 7
CLICK_THRESHOLD = 0.055      # normalized distance between thumb tip and index tip
CLICK_COOLDOWN = 0.35        # seconds
MOVE_GESTURE_MIN_DIST = 0.04 # normalized distance between index and middle
MODEL_PATH = "models/hand_landmarker.task"   # put your downloaded model here
# ------------------------------------------q

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

last_click_time = 0
click_active = False


def norm_distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def to_pixel_coords(lm, frame_w, frame_h):
    return int(lm.x * frame_w), int(lm.y * frame_h)


def is_cursor_gesture_valid(index_tip, middle_tip):
    dist_im = norm_distance(index_tip, middle_tip)
    return dist_im > MOVE_GESTURE_MIN_DIST


def draw_hand_points(frame, landmarks, frame_w, frame_h):
    # optional simple drawing without mp.solutions.drawing_utils
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    for a, b in connections:
        x1, y1 = to_pixel_coords(landmarks[a], frame_w, frame_h)
        x2, y2 = to_pixel_coords(landmarks[b], frame_w, frame_h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i, lm in enumerate(landmarks):
        x, y = to_pixel_coords(lm, frame_w, frame_h)
        cv2.circle(frame, (x, y), 4, (255, 0, 255), -1)


def main():
    global prev_x, prev_y, curr_x, curr_y
    global last_click_time, click_active

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # draw active area
            cv2.rectangle(
                frame,
                (FRAME_REDUCTION, FRAME_REDUCTION),
                (w - FRAME_REDUCTION, h - FRAME_REDUCTION),
                (255, 0, 255),
                2
            )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            status_text = "No Hand"

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                landmarks = result.hand_landmarks[0]

                draw_hand_points(frame, landmarks, w, h)

                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]

                thumb_px = to_pixel_coords(thumb_tip, w, h)
                index_px = to_pixel_coords(index_tip, w, h)
                middle_px = to_pixel_coords(middle_tip, w, h)

                cv2.circle(frame, thumb_px, 8, (255, 0, 0), -1)
                cv2.circle(frame, index_px, 8, (0, 255, 0), -1)
                cv2.circle(frame, middle_px, 8, (0, 0, 255), -1)

                # ---------- CURSOR MOVEMENT ----------
                if is_cursor_gesture_valid(index_tip, middle_tip):
                    cx = int((index_px[0] + middle_px[0]) / 2)
                    cy = int((index_px[1] + middle_px[1]) / 2)

                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)

                    clipped_x = max(FRAME_REDUCTION, min(cx, w - FRAME_REDUCTION))
                    clipped_y = max(FRAME_REDUCTION, min(cy, h - FRAME_REDUCTION))

                    screen_x = (clipped_x - FRAME_REDUCTION) * screen_w / (w - 2 * FRAME_REDUCTION)
                    screen_y = (clipped_y - FRAME_REDUCTION) * screen_h / (h - 2 * FRAME_REDUCTION)

                    curr_x = prev_x + (screen_x - prev_x) / SMOOTHING
                    curr_y = prev_y + (screen_y - prev_y) / SMOOTHING

                    pyautogui.moveTo(curr_x, curr_y)

                    prev_x, prev_y = curr_x, curr_y
                    status_text = "Cursor Move: ON"
                else:
                    status_text = "Cursor Move: OFF"

                # ---------- LEFT CLICK ----------
                thumb_index_dist = norm_distance(thumb_tip, index_tip)

                cv2.line(frame, thumb_px, index_px, (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"TI Dist: {thumb_index_dist:.3f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                current_time = time.time()

                if thumb_index_dist < CLICK_THRESHOLD:
                    pinch_mid = ((thumb_px[0] + index_px[0]) // 2, (thumb_px[1] + index_px[1]) // 2)
                    cv2.circle(frame, pinch_mid, 10, (0, 255, 255), -1)

                    if (not click_active) and (current_time - last_click_time > CLICK_COOLDOWN):
                        pyautogui.click()
                        click_active = True
                        last_click_time = current_time
                        status_text = "Left Click"
                else:
                    click_active = False

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Gesture Mouse Control - New MediaPipe", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()