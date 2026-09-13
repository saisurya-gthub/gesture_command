import cv2
import time
import math
import pyautogui
import mediapipe as mp
from mediapipe.tasks.python import vision

# =========================================================
# MODEL / MEDIAPIPE CONFIGURATION
# =========================================================
MODEL_PATH = "models/hand_landmarker.task"

# =========================================================
# CAMERA CONFIGURATION
# =========================================================
CAM_W = 1280
CAM_H = 720
FPS = 30.0

# =========================================================
# ACTIVE REGION CONFIGURATION
# Using the same cursor region settings as cursor_control_3.py
# =========================================================
FRAME_MARGIN_X = 80
FRAME_MARGIN_Y = 35

# =========================================================
# CURSOR MOVEMENT CONFIGURATION
# Using the same cursor movement settings as cursor_control_3.py
# =========================================================
SMOOTHING = 0.80
SPEED_FACTOR = 0.50

# If the hand is temporarily lost, reuse the last landmarks
MAX_MISSING_FRAMES = 12

# =========================================================
# TWO-FINGER CURSOR ACTIVATION CONFIGURATION
# Replaced with the simpler fixed-threshold logic from cursor_control_3.py
# index tip = 8, middle tip = 12
# =========================================================
ON_THRESHOLD_812 = 0.05
OFF_THRESHOLD_812 = 0.05

# Middle finger is naturally a little longer than index finger,
# so allow a small vertical difference between their tips.
VERTICAL_Y_TOLERANCE_ON = 0.050
VERTICAL_Y_TOLERANCE_OFF = 0.075

TOUCH_STABLE_FRAMES = 2
RELEASE_STABLE_FRAMES = 3

# =========================================================
# LEFT CLICK / DOUBLE CLICK / DRAG CONFIGURATION
# =========================================================
CLICK_THRESHOLD_48 = 0.075
CLICK_STABLE_FRAMES = 2
CLICK_COOLDOWN = 0.25

DOUBLE_CLICK_THRESHOLD = 0.65
DRAG_HOLD_TIME = 0.55
RELEASE_MARGIN = 0.025

# =========================================================
# RIGHT CLICK CONFIGURATION
# =========================================================
RIGHT_CLICK_THRESHOLD_412 = 0.080
RIGHT_CLICK_STABLE_FRAMES = 2
RIGHT_CLICK_COOLDOWN = 0.30

# =========================================================
# SCROLL CONFIGURATION
# =========================================================
POSE_CONFIRM_FRAMES = 3
SCROLL_ACTIVATE_FRAMES = 2

UPWARD_FINGER_THRESHOLD = 0.018
DOWNWARD_FINGER_THRESHOLD = 0.014

SCROLL_AMOUNT = 65
MAX_SCROLL_STEP_PER_FRAME = 2
VERTICAL_DOMINANCE_RATIO = 0.85
RESET_GRACE_FRAMES = 6
ANCHOR_SETTLE_FRAMES = 3
SCROLL_SMOOTHING_ALPHA = 0.35

# =========================================================
# PYAUTOGUI CONFIGURATION
# =========================================================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

SCREEN_W, SCREEN_H = pyautogui.size()

# =========================================================
# GLOBAL STATE VARIABLES
# =========================================================
prev_x, prev_y = 0, 0

cursor_active = False
touch_count = 0
release_count = 0

last_timestamp_ms = 0

click_count = 0
last_click_time = 0.0
click_ready = True

last_single_click_time = 0.0
pending_single_click = False
pending_click_time = 0.0

is_dragging = False
pinch_start_time = 0.0

right_click_count = 0
last_right_click_time = 0.0
right_click_ready = True

pose_confirm_frames = 0
stable_index_only_frames = 0
scroll_anchor_y = None
scroll_anchor_x = None
scroll_mode_active = False
missing_gesture_frames = 0
filtered_index_x = None
filtered_index_y = None
anchor_settle_counter = 0

# =========================================================
# MEDIAPIPE TASK SETUP
# =========================================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


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


def smooth_scroll_point(curr_x, curr_y):
    global filtered_index_x, filtered_index_y

    if filtered_index_x is None or filtered_index_y is None:
        filtered_index_x = curr_x
        filtered_index_y = curr_y
    else:
        filtered_index_x = (
            SCROLL_SMOOTHING_ALPHA * curr_x +
            (1 - SCROLL_SMOOTHING_ALPHA) * filtered_index_x
        )
        filtered_index_y = (
            SCROLL_SMOOTHING_ALPHA * curr_y +
            (1 - SCROLL_SMOOTHING_ALPHA) * filtered_index_y
        )

    return filtered_index_x, filtered_index_y


def is_finger_clearly_up(landmarks, tip_idx, pip_idx, mcp_idx, margin=0.015):
    tip_y = landmarks[tip_idx].y
    pip_y = landmarks[pip_idx].y
    mcp_y = landmarks[mcp_idx].y
    return tip_y < pip_y - margin and tip_y < mcp_y - margin


def is_finger_strongly_closed(landmarks, tip_idx, dip_idx, pip_idx, mcp_idx, margin=0.01):
    tip_y = landmarks[tip_idx].y
    dip_y = landmarks[dip_idx].y
    pip_y = landmarks[pip_idx].y
    mcp_y = landmarks[mcp_idx].y

    return (
        tip_y > pip_y + margin and
        tip_y > dip_y + margin * 0.5 and
        tip_y > mcp_y - margin
    )


def is_index_only_pose_strict(landmarks):
    index_up = is_finger_clearly_up(landmarks, 8, 6, 5, margin=0.012)

    middle_closed = is_finger_strongly_closed(landmarks, 12, 11, 10, 9, margin=0.008)
    ring_closed = is_finger_strongly_closed(landmarks, 16, 15, 14, 13, margin=0.008)
    pinky_closed = is_finger_strongly_closed(landmarks, 20, 19, 18, 17, margin=0.008)

    return index_up and middle_closed and ring_closed and pinky_closed


def reset_scroll_state():
    global pose_confirm_frames
    global stable_index_only_frames
    global scroll_anchor_x, scroll_anchor_y
    global scroll_mode_active
    global missing_gesture_frames
    global filtered_index_x, filtered_index_y
    global anchor_settle_counter

    pose_confirm_frames = 0
    stable_index_only_frames = 0
    scroll_anchor_x = None
    scroll_anchor_y = None
    scroll_mode_active = False
    missing_gesture_frames = 0
    filtered_index_x = None
    filtered_index_y = None
    anchor_settle_counter = 0


# =========================================================
# REPLACED CURSOR ACTIVATION LOGIC
# This is now the same style as cursor_control_3.py
# =========================================================
def update_two_finger_state(index_tip, middle_tip):
    global cursor_active, touch_count, release_count

    d_im = distance_2d(index_tip, middle_tip)

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


def detect_click_drag_actions(thumb_tip, index_tip, action_allowed):
    global click_count, last_click_time, click_ready
    global last_single_click_time
    global is_dragging, pinch_start_time
    global pending_single_click, pending_click_time

    d_ti = distance_2d(thumb_tip, index_tip)
    now = time.time()

    clicked = False
    double_clicked = False
    drag_started = False
    dragging = is_dragging
    dropped = False

    if pending_single_click and (now - pending_click_time) > DOUBLE_CLICK_THRESHOLD:
        pyautogui.click(button="left")
        clicked = True
        pending_single_click = False
        last_click_time = now

    if is_dragging:
        if d_ti > (CLICK_THRESHOLD_48 + RELEASE_MARGIN):
            pyautogui.mouseUp()
            is_dragging = False
            dragging = False
            dropped = True
            click_ready = True
            pinch_start_time = 0.0
        return clicked, double_clicked, drag_started, dragging, dropped, d_ti

    if action_allowed and d_ti < CLICK_THRESHOLD_48:
        click_count += 1
    else:
        click_count = 0

    if action_allowed and click_ready and click_count >= CLICK_STABLE_FRAMES:
        click_ready = False
        pinch_start_time = now

    if not click_ready and action_allowed:
        if d_ti < CLICK_THRESHOLD_48 and (now - pinch_start_time) >= DRAG_HOLD_TIME:
            pending_single_click = False
            pyautogui.mouseDown()
            is_dragging = True
            dragging = True
            drag_started = True

    if d_ti > (CLICK_THRESHOLD_48 + RELEASE_MARGIN):
        if not click_ready:
            pinch_duration = now - pinch_start_time

            if action_allowed and not is_dragging and pinch_duration < DRAG_HOLD_TIME:
                if pending_single_click and (now - pending_click_time) <= DOUBLE_CLICK_THRESHOLD:
                    pyautogui.doubleClick()
                    double_clicked = True
                    pending_single_click = False
                    last_single_click_time = 0.0
                    last_click_time = now
                else:
                    pending_single_click = True
                    pending_click_time = now
                    last_single_click_time = now

        click_ready = True
        pinch_start_time = 0.0

    return clicked, double_clicked, drag_started, dragging, dropped, d_ti


def detect_right_click(thumb_tip, middle_tip, action_allowed):
    global right_click_count, last_right_click_time, right_click_ready
    global is_dragging

    d_tm = distance_2d(thumb_tip, middle_tip)
    now = time.time()
    right_clicked = False

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
# SAME CURSOR MOVEMENT MAPPING AS cursor_control_3.py
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


def process_scroll(hand_landmarks):
    global pose_confirm_frames
    global stable_index_only_frames
    global scroll_anchor_x, scroll_anchor_y
    global scroll_mode_active
    global missing_gesture_frames
    global anchor_settle_counter

    strict_pose = is_index_only_pose_strict(hand_landmarks)

    raw_index_x = hand_landmarks[8].x
    raw_index_y = hand_landmarks[8].y
    curr_index_x, curr_index_y = smooth_scroll_point(raw_index_x, raw_index_y)

    if not strict_pose:
        missing_gesture_frames += 1
        pose_confirm_frames = 0
        stable_index_only_frames = 0

        if missing_gesture_frames >= RESET_GRACE_FRAMES:
            reset_scroll_state()
            return "IDLE"

        return "WAITING POSE"

    missing_gesture_frames = 0
    pose_confirm_frames += 1

    if pose_confirm_frames < POSE_CONFIRM_FRAMES:
        return "CONFIRMING POSE"

    stable_index_only_frames += 1

    if stable_index_only_frames < SCROLL_ACTIVATE_FRAMES:
        return "SCROLL READY"

    if not scroll_mode_active:
        scroll_mode_active = True
        scroll_anchor_x = curr_index_x
        scroll_anchor_y = curr_index_y
        anchor_settle_counter = 0
        return "ANCHOR SET"

    if anchor_settle_counter < ANCHOR_SETTLE_FRAMES:
        scroll_anchor_x = curr_index_x
        scroll_anchor_y = curr_index_y
        anchor_settle_counter += 1
        return "ANCHOR SETTLING"

    dx_from_anchor = curr_index_x - scroll_anchor_x
    dy_from_anchor = curr_index_y - scroll_anchor_y

    if abs(dy_from_anchor) < abs(dx_from_anchor) * VERTICAL_DOMINANCE_RATIO:
        scroll_anchor_x = 0.7 * scroll_anchor_x + 0.3 * curr_index_x
        scroll_anchor_y = 0.7 * scroll_anchor_y + 0.3 * curr_index_y
        return "SCROLL MODE"

    if dy_from_anchor <= -UPWARD_FINGER_THRESHOLD:
        steps = min(
            max(1, int(abs(dy_from_anchor) / UPWARD_FINGER_THRESHOLD)),
            MAX_SCROLL_STEP_PER_FRAME
        )
        pyautogui.scroll(-SCROLL_AMOUNT * steps)
        scroll_anchor_x = 0.85 * curr_index_x + 0.15 * scroll_anchor_x
        scroll_anchor_y = 0.85 * curr_index_y + 0.15 * scroll_anchor_y
        return f"SCROLL DOWN x{steps}"

    elif dy_from_anchor >= DOWNWARD_FINGER_THRESHOLD:
        steps = min(
            max(1, int(abs(dy_from_anchor) / DOWNWARD_FINGER_THRESHOLD)),
            MAX_SCROLL_STEP_PER_FRAME
        )
        pyautogui.scroll(SCROLL_AMOUNT * steps)
        scroll_anchor_x = 0.85 * curr_index_x + 0.15 * scroll_anchor_x
        scroll_anchor_y = 0.85 * curr_index_y + 0.15 * scroll_anchor_y
        return f"SCROLL UP x{steps}"

    scroll_anchor_x = 0.8 * scroll_anchor_x + 0.2 * curr_index_x
    scroll_anchor_y = 0.8 * scroll_anchor_y + 0.2 * curr_index_y

    return "SCROLL MODE"


def main():
    global prev_x, prev_y
    global missing_gesture_frames
    global cursor_active, touch_count, release_count
    global click_count, right_click_count, pending_single_click

    missing_hand_frames = 0
    last_valid_landmarks = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

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

            move_text = "Cursor Move: OFF"
            action_text = "Action Triggered: None"
            scroll_text = "Scroll: OFF"
            status_text = "No Hand"
            d_im = 0.0
            y_diff = 0.0
            d_ti = 0.0
            d_tm = 0.0
            action_allowed = False

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
                    status_text = "Hand Partially Lost - Using Memory"
                else:
                    hand_available = False
                    status_text = "Tracking Lost - Show More Hand"

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

                cv2.line(frame, (index_px, index_py), (middle_px, middle_py), (255, 0, 255), 2)
                cv2.line(frame, (thumb_px, thumb_py), (index_px, index_py), (0, 255, 255), 2)
                cv2.line(frame, (thumb_px, thumb_py), (middle_px, middle_py), (255, 165, 0), 2)

                cv2.circle(frame, (thumb_px, thumb_py), 8, (255, 0, 0), -1)
                cv2.circle(frame, (index_px, index_py), 8, (0, 255, 255), -1)
                cv2.circle(frame, (middle_px, middle_py), 8, (255, 255, 0), -1)

                scroll_state = process_scroll(hand_landmarks)
                scroll_active_now = scroll_state not in {"IDLE", "WAITING POSE"}

                if scroll_active_now:
                    scroll_text = f"Scroll: {scroll_state}"
                    move_text = "Cursor Move: OFF (Scroll Active)"
                    action_text = "Action Triggered: None"

                    cursor_active = False
                    touch_count = 0
                    release_count = 0

                    click_count = 0
                    right_click_count = 0
                    pending_single_click = False

                    if scroll_anchor_x is not None and scroll_anchor_y is not None:
                        ax = int(scroll_anchor_x * CAM_W)
                        ay = int(scroll_anchor_y * CAM_H)
                        cv2.circle(frame, (ax, ay), 8, (0, 0, 255), -1)
                        cv2.putText(
                            frame,
                            "ANCHOR",
                            (ax + 10, ay),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2
                        )
                else:
                    scroll_text = "Scroll: OFF"

                    can_move, d_im, y_diff = update_two_finger_state(index_tip, middle_tip)

                    action_allowed = hand_available

                    if can_move:
                        cam_x, cam_y, _, _ = move_cursor(index_tip)
                        cv2.circle(frame, (cam_x, cam_y), 8, (0, 0, 255), -1)

                    clicked, double_clicked, drag_started, dragging, dropped, d_ti = detect_click_drag_actions(
                        thumb_tip, index_tip, action_allowed
                    )

                    right_clicked, d_tm = detect_right_click(thumb_tip, middle_tip, action_allowed)

                    if dragging:
                        move_text = "Dragging"
                    elif can_move:
                        move_text = "Cursor Move: ON"
                    else:
                        move_text = "Cursor Move: OFF"

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

            else:
                missing_gesture_frames += 1
                if missing_gesture_frames >= RESET_GRACE_FRAMES:
                    reset_scroll_state()
                scroll_text = "Scroll: OFF"

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

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

            cv2.putText(frame, scroll_text, (10, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            cv2.putText(frame, f"8-12 Dist: {d_im:.4f}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"Y Diff: {y_diff:.4f}", (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"4-8 Dist (Left/Drag): {d_ti:.4f}", (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, f"4-12 Dist (Right): {d_tm:.4f}", (10, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            cv2.putText(frame, f"Action Allowed: {action_allowed}", (10, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.imshow("Gesture Cursor + Scroll Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
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