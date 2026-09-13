import cv2
import time
import math
import pyautogui
import mediapipe as mp
from mediapipe.tasks.python import vision

# =========================================================
# MODEL / MEDIAPIPE CONFIGURATION
# =========================================================
# Path to the MediaPipe hand landmarker model
MODEL_PATH = "models/hand_landmarker.task"

# =========================================================
# CAMERA CONFIGURATION
# =========================================================
CAM_W = 1280
CAM_H = 720

# =========================================================
# ACTIVE REGION CONFIGURATION
# =========================================================
# Cursor movement is restricted within this inner region
# to improve stability and reduce edge jitter
FRAME_MARGIN_X = 80
FRAME_MARGIN_Y = 60

# =========================================================
# CURSOR MOVEMENT CONFIGURATION
# =========================================================
# Higher SMOOTHING -> smoother but slower cursor
SMOOTHING = 0.80

# Higher SPEED_FACTOR -> faster response
SPEED_FACTOR = 0.35

# =========================================================
# HAND LOSS MEMORY CONFIGURATION
# =========================================================
# If the hand is temporarily lost, reuse the last landmarks
# for a few frames to avoid sudden OFF flicker
MAX_MISSING_FRAMES = 12

# =========================================================
# TWO-FINGER CURSOR ACTIVATION CONFIGURATION
# =========================================================
# Cursor movement is enabled when index tip (8) and
# middle tip (12) are close and vertically aligned
ON_THRESHOLD_812 = 0.050
OFF_THRESHOLD_812 = 0.050

# Middle finger is usually slightly longer than index finger,
# so allow a small vertical tip difference
VERTICAL_Y_TOLERANCE_ON = 0.050
VERTICAL_Y_TOLERANCE_OFF = 0.075

# Stable frames needed to switch movement ON/OFF
TOUCH_STABLE_FRAMES = 2
RELEASE_STABLE_FRAMES = 3

# =========================================================
# LEFT CLICK / DOUBLE CLICK / DRAG CONFIGURATION
# =========================================================
# Thumb tip (4) + Index tip (8)
CLICK_THRESHOLD_48 = 0.065
CLICK_STABLE_FRAMES = 2
CLICK_COOLDOWN = 0.25

# Time window to interpret two quick single clicks as a double click
DOUBLE_CLICK_THRESHOLD = 0.40

# Hold time to convert pinch into drag instead of normal click
DRAG_HOLD_TIME = 0.45

# Release hysteresis for safer reset
RELEASE_MARGIN = 0.020

# =========================================================
# RIGHT CLICK CONFIGURATION
# =========================================================
# Thumb tip (4) + Middle tip (12)
RIGHT_CLICK_THRESHOLD_412 = 0.070
RIGHT_CLICK_STABLE_FRAMES = 2
RIGHT_CLICK_COOLDOWN = 0.25

# =========================================================
# PYAUTOGUI CONFIGURATION
# =========================================================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

SCREEN_W, SCREEN_H = pyautogui.size()

# =========================================================
# GLOBAL STATE VARIABLES
# =========================================================
# Previous cursor position for smoothing
prev_x, prev_y = 0, 0

# Cursor movement gesture state
cursor_active = False

# Stable counters for movement gesture
touch_count = 0
release_count = 0

# Timestamp safety for MediaPipe VIDEO mode
last_timestamp_ms = 0

# Left click / double click / drag state
click_count = 0
last_click_time = 0.0
click_ready = True

# Track previous single click time for double click decision
last_single_click_time = 0.0

# Drag state
is_dragging = False
pinch_start_time = 0.0

# Right click state
right_click_count = 0
last_right_click_time = 0.0
right_click_ready = True

# =========================================================
# MEDIAPIPE TASK SETUP
# =========================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


# =========================================================
# FUNCTION: create_hand_landmarker
# Creates and returns the MediaPipe hand landmarker object
# =========================================================
def create_hand_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return HandLandmarker.create_from_options(options)


# =========================================================
# FUNCTION: distance_2d
# Computes Euclidean distance between two normalized landmarks
# =========================================================
def distance_2d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# =========================================================
# FUNCTION: map_range
# Maps one range to another
# =========================================================
def map_range(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    value = max(in_min, min(value, in_max))
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# =========================================================
# FUNCTION: get_monotonic_timestamp_ms
# Ensures strictly increasing timestamps for VIDEO mode
# =========================================================
def get_monotonic_timestamp_ms():
    global last_timestamp_ms
    now = int(time.time() * 1000)
    if now <= last_timestamp_ms:
        now = last_timestamp_ms + 1
    last_timestamp_ms = now
    return now


# =========================================================
# FUNCTION: update_two_finger_state
# Controls cursor movement ON/OFF using index and middle tips
# =========================================================
def update_two_finger_state(index_tip, middle_tip):
    global cursor_active, touch_count, release_count

    d_im = distance_2d(index_tip, middle_tip)

    # Vertical difference between middle and index fingertips
    y_diff = middle_tip.y - index_tip.y
    abs_y_diff = abs(y_diff)

    touching = (
        d_im < ON_THRESHOLD_812 and
        abs_y_diff < VERTICAL_Y_TOLERANCE_ON
    )

    separated = (
        d_im > OFF_THRESHOLD_812 or
        abs_y_diff > VERTICAL_Y_TOLERANCE_OFF
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

    return cursor_active, d_im, y_diff


# =========================================================
# FUNCTION: detect_click_drag_actions
# Handles:
# 1. Single left click
# 2. Double click
# 3. Drag start
# 4. Drop
#
# Logic:
# - Quick pinch/release -> click
# - Two quick click gestures -> double click
# - Long pinch hold -> drag
# - Release after drag -> drop
# =========================================================
def detect_click_drag_actions(thumb_tip, index_tip, action_allowed):
    global click_count, last_click_time, click_ready
    global last_single_click_time
    global is_dragging, pinch_start_time

    d_ti = distance_2d(thumb_tip, index_tip)
    now = time.time()

    clicked = False
    double_clicked = False
    drag_started = False
    dragging = is_dragging
    dropped = False

    # While dragging, do not allow normal click logic to fire again
    if is_dragging:
        if d_ti > (CLICK_THRESHOLD_48 + RELEASE_MARGIN):
            pyautogui.mouseUp()
            is_dragging = False
            dragging = False
            dropped = True
            click_ready = True
            pinch_start_time = 0.0
        return clicked, double_clicked, drag_started, dragging, dropped, d_ti

    # Track stable pinch frames
    if action_allowed and d_ti < CLICK_THRESHOLD_48:
        click_count += 1
    else:
        click_count = 0

    # Pinch entered
    if action_allowed and click_ready and click_count >= CLICK_STABLE_FRAMES:
        click_ready = False
        pinch_start_time = now

    # While pinch is held, convert to drag if held long enough
    if not click_ready and action_allowed:
        if d_ti < CLICK_THRESHOLD_48 and (now - pinch_start_time) >= DRAG_HOLD_TIME:
            pyautogui.mouseDown()
            is_dragging = True
            dragging = True
            drag_started = True

    # Pinch released
    if d_ti > (CLICK_THRESHOLD_48 + RELEASE_MARGIN):
        if not click_ready:
            # Treat as click gesture only if drag did not start
            if action_allowed and (now - last_click_time) > CLICK_COOLDOWN:
                if (now - last_single_click_time) < DOUBLE_CLICK_THRESHOLD:
                    pyautogui.doubleClick()
                    double_clicked = True
                    last_single_click_time = 0.0
                else:
                    pyautogui.click(button="left")
                    clicked = True
                    last_single_click_time = now

                last_click_time = now

        click_ready = True
        pinch_start_time = 0.0

    return clicked, double_clicked, drag_started, dragging, dropped, d_ti


# =========================================================
# FUNCTION: detect_right_click
# Handles right click using thumb + middle fingertip touch
# =========================================================
def detect_right_click(thumb_tip, middle_tip, action_allowed):
    global right_click_count, last_right_click_time, right_click_ready

    d_tm = distance_2d(thumb_tip, middle_tip)
    now = time.time()
    right_clicked = False

    # Disable right click while dragging to avoid conflicts
    if is_dragging:
        return right_clicked, d_tm

    if action_allowed and d_tm < RIGHT_CLICK_THRESHOLD_412:
        right_click_count += 1
    else:
        right_click_count = 0

    if action_allowed and right_click_ready and right_click_count >= RIGHT_CLICK_STABLE_FRAMES:
        if now - last_right_click_time > RIGHT_CLICK_COOLDOWN:
            pyautogui.click(button="right")
            last_right_click_time = now
            right_clicked = True
            right_click_ready = False

    if d_tm > (RIGHT_CLICK_THRESHOLD_412 + RELEASE_MARGIN):
        right_click_ready = True

    return right_clicked, d_tm


# =========================================================
# FUNCTION: move_cursor
# Moves the cursor using index fingertip with smoothing
# =========================================================
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


# =========================================================
# FUNCTION: main
# Main runtime loop for cursor movement, left click,
# right click, double click, drag and drop
# =========================================================
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

            # Default values for every frame
            move_text = "Cursor Move: OFF"
            action_text = "Action Triggered: None"
            status_text = "No Hand"
            d_im = 0.0
            y_diff = 0.0
            d_ti = 0.0
            d_tm = 0.0
            action_allowed = False

            # Draw active region
            cv2.rectangle(
                frame,
                (FRAME_MARGIN_X, FRAME_MARGIN_Y),
                (CAM_W - FRAME_MARGIN_X, CAM_H - FRAME_MARGIN_Y),
                (255, 255, 0),
                2
            )

            # Hand detection with memory support
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
                    status_text = "Hand Partially Lost - Using Memory"
                else:
                    hand_available = False
                    status_text = "Tracking Lost - Show More Hand"

            if hand_available:
                # Draw all landmarks
                for lm in hand_landmarks:
                    x = int(lm.x * CAM_W)
                    y = int(lm.y * CAM_H)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Important fingertips
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                middle_tip = hand_landmarks[12]

                thumb_px = int(thumb_tip.x * CAM_W)
                thumb_py = int(thumb_tip.y * CAM_H)

                index_px = int(index_tip.x * CAM_W)
                index_py = int(index_tip.y * CAM_H)

                middle_px = int(middle_tip.x * CAM_W)
                middle_py = int(middle_tip.y * CAM_H)

                # Draw index-middle relationship for cursor control
                cv2.line(frame, (index_px, index_py), (middle_px, middle_py), (255, 0, 255), 2)
                cv2.circle(frame, (index_px, index_py), 8, (0, 255, 255), -1)
                cv2.circle(frame, (middle_px, middle_py), 8, (255, 255, 0), -1)

                # Draw thumb point
                cv2.circle(frame, (thumb_px, thumb_py), 8, (255, 0, 0), -1)

                # Cursor movement state
                can_move, d_im, y_diff = update_two_finger_state(index_tip, middle_tip)

                # Allow click actions when hand is available
                action_allowed = hand_available

                # Move cursor only when movement gesture is active
                if can_move:
                    cam_x, cam_y, _, _ = move_cursor(index_tip)
                    cv2.circle(frame, (cam_x, cam_y), 8, (0, 0, 255), -1)

                # Left click / double click / drag / drop
                clicked, double_clicked, drag_started, dragging, dropped, d_ti = detect_click_drag_actions(
                    thumb_tip, index_tip, action_allowed
                )

                # Right click
                right_clicked, d_tm = detect_right_click(thumb_tip, middle_tip, action_allowed)

                # Draw gesture lines
                cv2.line(frame, (thumb_px, thumb_py), (index_px, index_py), (0, 255, 255), 2)
                cv2.line(frame, (thumb_px, thumb_py), (middle_px, middle_py), (255, 165, 0), 2)

                # Movement text
                if dragging:
                    move_text = "Dragging"
                elif can_move:
                    move_text = "Cursor Move: ON"
                else:
                    move_text = "Cursor Move: OFF"

                # Action priority
                if drag_started:
                    action_text = "Action Triggered: Drag Start"
                elif dropped:
                    action_text = "Action Triggered: Drop"
                elif double_clicked:
                    action_text = "Action Triggered: Double Click"
                elif right_clicked:
                    action_text = "Action Triggered: Right Click"
                elif clicked:
                    action_text = "Action Triggered: Left Click"

            # FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # On-screen text
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, status_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, "Press Q to quit", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            cv2.putText(frame, move_text, (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, action_text, (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"8-12 Dist: {d_im:.4f}", (10, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"Y Diff: {y_diff:.4f}", (10, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"4-8 Dist (Left/Drag): {d_ti:.4f}", (10, 265),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, f"4-12 Dist (Right): {d_tm:.4f}", (10, 295),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            cv2.putText(frame, f"Action Allowed: {action_allowed}", (10, 325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("Gesture Cursor Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Runtime error:", e)
        traceback.print_exc()
        input("Press Enter to exit...")