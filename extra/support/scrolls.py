import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# =========================
# Camera / Frame Settings
# =========================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30.0

# =========================
# Scroll Settings
# =========================
# =========================
# Scroll Settings
# =========================
POSE_CONFIRM_FRAMES = 3
SCROLL_ACTIVATE_FRAMES = 2

# Separate thresholds for both directions
UPWARD_FINGER_THRESHOLD = 0.018     # finger goes up -> screen goes down
DOWNWARD_FINGER_THRESHOLD = 0.014   # finger goes down -> screen goes up

SCROLL_AMOUNT = 65
MAX_STEP_PER_FRAME = 2
VERTICAL_DOMINANCE_RATIO = 0.85
RESET_GRACE_FRAMES = 6
ANCHOR_SETTLE_FRAMES = 3
SMOOTHING_ALPHA = 0.35

# =========================
# Global State
# =========================
pose_confirm_frames = 0
stable_index_only_frames = 0
scroll_anchor_y = None
scroll_anchor_x = None
scroll_mode_active = False
missing_gesture_frames = 0
filtered_index_x = None
filtered_index_y = None
anchor_settle_counter = 0


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


def smooth_point(curr_x, curr_y):
    global filtered_index_x, filtered_index_y

    if filtered_index_x is None or filtered_index_y is None:
        filtered_index_x = curr_x
        filtered_index_y = curr_y
    else:
        filtered_index_x = (
            SMOOTHING_ALPHA * curr_x + (1 - SMOOTHING_ALPHA) * filtered_index_x
        )
        filtered_index_y = (
            SMOOTHING_ALPHA * curr_y + (1 - SMOOTHING_ALPHA) * filtered_index_y
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

    # Strongly closed means fingertip is not just lower than PIP,
    # but also lower than DIP and near/below MCP region.
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
    curr_index_x, curr_index_y = smooth_point(raw_index_x, raw_index_y)

    if not strict_pose:
        missing_gesture_frames += 1

        # Break pose confirmation quickly if fingers reopen
        pose_confirm_frames = 0
        stable_index_only_frames = 0

        if missing_gesture_frames >= RESET_GRACE_FRAMES:
            reset_scroll_state()
            return "IDLE"

        return "WAITING POSE"

    missing_gesture_frames = 0
    pose_confirm_frames += 1

    # Do not set anchor until pose is confirmed for consecutive frames
    if pose_confirm_frames < POSE_CONFIRM_FRAMES:
        return "CONFIRMING POSE"

    stable_index_only_frames += 1

    # After pose confirmation, wait a couple of frames before scroll mode
    if stable_index_only_frames < SCROLL_ACTIVATE_FRAMES:
        return "SCROLL READY"

    # Set anchor only once AFTER pose has been confirmed
    if not scroll_mode_active:
        scroll_mode_active = True
        scroll_anchor_x = curr_index_x
        scroll_anchor_y = curr_index_y
        anchor_settle_counter = 0
        return "ANCHOR SET"

    # Let anchor settle for a few frames
    if anchor_settle_counter < ANCHOR_SETTLE_FRAMES:
        scroll_anchor_x = curr_index_x
        scroll_anchor_y = curr_index_y
        anchor_settle_counter += 1
        return "ANCHOR SETTLING"

    dx_from_anchor = curr_index_x - scroll_anchor_x
    dy_from_anchor = curr_index_y - scroll_anchor_y

    # Require mostly vertical movement
    if abs(dy_from_anchor) < abs(dx_from_anchor) * VERTICAL_DOMINANCE_RATIO:
        scroll_anchor_x = 0.7 * scroll_anchor_x + 0.3 * curr_index_x
        scroll_anchor_y = 0.7 * scroll_anchor_y + 0.3 * curr_index_y
        return "SCROLL MODE"

    # Finger moves UP -> screen scrolls DOWN
    # Finger moves UP -> screen scrolls DOWN
    if dy_from_anchor <= -UPWARD_FINGER_THRESHOLD:
        steps = min(
            max(1, int(abs(dy_from_anchor) / UPWARD_FINGER_THRESHOLD)),
            MAX_STEP_PER_FRAME
        )
        pyautogui.scroll(-SCROLL_AMOUNT * steps)
        scroll_anchor_x = 0.85 * curr_index_x + 0.15 * scroll_anchor_x
        scroll_anchor_y = 0.85 * curr_index_y + 0.15 * scroll_anchor_y
        return f"SCROLL DOWN x{steps}"

    # Finger moves DOWN -> screen scrolls UP
    elif dy_from_anchor >= DOWNWARD_FINGER_THRESHOLD:
        steps = min(
            max(1, int(abs(dy_from_anchor) / DOWNWARD_FINGER_THRESHOLD)),
            MAX_STEP_PER_FRAME
        )
        pyautogui.scroll(SCROLL_AMOUNT * steps)
        scroll_anchor_x = 0.85 * curr_index_x + 0.15 * scroll_anchor_x
        scroll_anchor_y = 0.85 * curr_index_y + 0.15 * scroll_anchor_y
        return f"SCROLL UP x{steps}"

    # Slow anchor drift in dead zone
    scroll_anchor_x = 0.8 * scroll_anchor_x + 0.2 * curr_index_x
    scroll_anchor_y = 0.8 * scroll_anchor_y + 0.2 * curr_index_y

    return "SCROLL MODE"


def main():
    global missing_gesture_frames

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera resolution: {actual_width} x {actual_height}")
    print(f"Camera FPS: {actual_fps:.2f}")

    with create_hand_landmarker() as landmarker:
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int((frame_index / FPS) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            status_text = "NO HAND"

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                for lm in hand_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                status_text = process_scroll(hand_landmarks)

                if scroll_anchor_x is not None and scroll_anchor_y is not None:
                    ax = int(scroll_anchor_x * frame.shape[1])
                    ay = int(scroll_anchor_y * frame.shape[0])
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
                missing_gesture_frames += 1
                if missing_gesture_frames >= RESET_GRACE_FRAMES:
                    reset_scroll_state()
                status_text = "NO HAND"

            cv2.putText(
                frame,
                f"Status: {status_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Index only -> Scroll | Q -> Quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("MediaPipe Tasks API - Smooth Scroll Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()