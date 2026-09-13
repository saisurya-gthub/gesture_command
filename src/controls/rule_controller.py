import time
import math
import pyautogui

# Camera size
CAM_W = 1280
CAM_H = 720

# Active cursor area
FRAME_MARGIN_X = 80
FRAME_MARGIN_Y = 35

# Cursor smoothing
SMOOTHING = 0.80
SPEED_FACTOR = 0.50

# Cursor activation thresholds
ON_THRESHOLD_812 = 0.050
OFF_THRESHOLD_812 = 0.065
VERTICAL_Y_TOLERANCE_ON = 0.050
VERTICAL_Y_TOLERANCE_OFF = 0.075
TOUCH_STABLE_FRAMES = 2
RELEASE_STABLE_FRAMES = 3

# Click / drag thresholds
CLICK_THRESHOLD_48 = 0.075
CLICK_STABLE_FRAMES = 2
DOUBLE_CLICK_THRESHOLD = 0.65
DRAG_HOLD_TIME = 0.55
RELEASE_MARGIN = 0.025

# Right click thresholds
RIGHT_CLICK_THRESHOLD_412 = 0.080
RIGHT_CLICK_STABLE_FRAMES = 2
RIGHT_CLICK_COOLDOWN = 0.30

# Scroll thresholds
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

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

SCREEN_W, SCREEN_H = pyautogui.size()

# Cursor state
prev_x, prev_y = SCREEN_W // 2, SCREEN_H // 2
cursor_active = False
touch_count = 0
release_count = 0

# Click state
click_count = 0
click_ready = True
pending_single_click = False
pending_click_time = 0.0
is_dragging = False
pinch_start_time = 0.0

# Right click state
right_click_count = 0
last_right_click_time = 0.0
right_click_ready = True

# Scroll state
pose_confirm_frames = 0
stable_index_only_frames = 0
scroll_anchor_x = None
scroll_anchor_y = None
scroll_mode_active = False
missing_gesture_frames = 0
filtered_index_x = None
filtered_index_y = None
anchor_settle_counter = 0


def distance_2d(a, b):
    """Return 2D distance between two landmarks."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def map_range(value, in_min, in_max, out_min, out_max):
    """Map value from one range to another."""
    if in_max == in_min:
        return out_min
    value = max(in_min, min(value, in_max))
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def smooth_scroll_point(curr_x, curr_y):
    """Smooth index point for scroll."""
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
    """Check if finger is clearly up."""
    tip_y = landmarks[tip_idx].y
    pip_y = landmarks[pip_idx].y
    mcp_y = landmarks[mcp_idx].y
    return tip_y < pip_y - margin and tip_y < mcp_y - margin


def is_finger_strongly_closed(landmarks, tip_idx, dip_idx, pip_idx, mcp_idx, margin=0.01):
    """Check if finger is strongly closed."""
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
    """Check strict index-only pose for scroll."""
    index_up = is_finger_clearly_up(landmarks, 8, 6, 5, margin=0.012)
    middle_closed = is_finger_strongly_closed(landmarks, 12, 11, 10, 9, margin=0.008)
    ring_closed = is_finger_strongly_closed(landmarks, 16, 15, 14, 13, margin=0.008)
    pinky_closed = is_finger_strongly_closed(landmarks, 20, 19, 18, 17, margin=0.008)

    return index_up and middle_closed and ring_closed and pinky_closed


def reset_scroll_state():
    """Reset scroll-only state."""
    global pose_confirm_frames, stable_index_only_frames
    global scroll_anchor_x, scroll_anchor_y
    global scroll_mode_active, missing_gesture_frames
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


def reset_rule_state():
    """Reset rule-based state when hand is lost."""
    global cursor_active, touch_count, release_count
    global click_count, click_ready, pending_single_click, pending_click_time
    global is_dragging, pinch_start_time
    global right_click_count, right_click_ready
    global filtered_index_x, filtered_index_y

    cursor_active = False
    touch_count = 0
    release_count = 0

    click_count = 0
    click_ready = True
    pending_single_click = False
    pending_click_time = 0.0

    if is_dragging:
        pyautogui.mouseUp()
    is_dragging = False
    pinch_start_time = 0.0

    right_click_count = 0
    right_click_ready = True

    filtered_index_x = None
    filtered_index_y = None
    reset_scroll_state()


def update_two_finger_state(index_tip, middle_tip):
    """Detect cursor move activation."""
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


def move_cursor(index_tip):
    """Move cursor using index tip."""
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
    return curr_x, curr_y


def detect_click_drag_actions(thumb_tip, index_tip, action_allowed):
    """Detect left click, double click, drag and drop."""
    global click_count, click_ready
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
                else:
                    pending_single_click = True
                    pending_click_time = now

        click_ready = True
        pinch_start_time = 0.0

    return clicked, double_clicked, drag_started, dragging, dropped, d_ti


def detect_right_click(thumb_tip, middle_tip, action_allowed):
    """Detect right click using thumb and middle finger."""
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


def process_scroll(hand_landmarks):
    """Handle index-only scroll."""
    global pose_confirm_frames, stable_index_only_frames
    global scroll_anchor_x, scroll_anchor_y
    global scroll_mode_active, missing_gesture_frames
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

    if dy_from_anchor >= DOWNWARD_FINGER_THRESHOLD:
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


def handle_rule_based(hand_landmarks, rules_enabled=True):
    """Run rule-based actions and report status."""
    if not rules_enabled:
        return False, "Frozen"

    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    thumb_tip = hand_landmarks[4]

    scroll_state = process_scroll(hand_landmarks)
    if scroll_state not in ["IDLE", "WAITING POSE"]:
        return True, scroll_state

    can_move, _, _ = update_two_finger_state(index_tip, middle_tip)
    if can_move:
        move_cursor(index_tip)
        return True, "Cursor Move"

    clicked, double_clicked, drag_started, dragging, dropped, _ = detect_click_drag_actions(
        thumb_tip, index_tip, True
    )

    if drag_started:
        return True, "Drag Start"
    if dragging:
        return True, "Dragging"
    if dropped:
        return True, "Drop"
    if double_clicked:
        return True, "Double Click"
    if clicked:
        return True, "Left Click"

    right_clicked, _ = detect_right_click(thumb_tip, middle_tip, True)
    if right_clicked:
        return True, "Right Click"

    return False, "None"