import cv2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_landmarks(frame, landmarks):
    """Draw hand landmarks and connections."""
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
    """Draw one info line on frame."""
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
    """Draw top-right mode banner."""
    mode = state.get("mode", "UNKNOWN")
    pending_action = state.get("pending_action", None)

    if mode == "CONFIRMATION":
        banner_color = (0, 0, 255)
    elif mode == "BROWSER":
        banner_color = (255, 180, 0)
    elif mode == "PRESENTATION":
        banner_color = (255, 0, 255)
    else:
        banner_color = (0, 180, 255)

    h, w, _ = frame.shape
    box_w = 290
    box_h = 45
    x1 = w - box_w - 10
    y1 = 10
    x2 = w - 10
    y2 = y1 + box_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), banner_color, -1)
    cv2.putText(
        frame,
        f"MODE: {mode}",
        (x1 + 10, y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if pending_action:
        cv2.rectangle(frame, (x1, y2 + 8), (x2, y2 + 43), (50, 50, 50), -1)
        cv2.putText(
            frame,
            f"PENDING: {pending_action}",
            (x1 + 10, y2 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


def draw_confirmation_overlay(frame, state):
    """Draw large confirmation box."""
    if state.get("mode") != "CONFIRMATION":
        return

    h, w, _ = frame.shape
    x1, y1 = 40, h - 155
    x2, y2 = w - 40, h - 30

    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    pending_action = state.get("pending_action") or "ACTION"

    cv2.putText(
        frame,
        f"{pending_action} selected - Confirmation Mode",
        (x1 + 15, y1 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "thumbs_up  -> Proceed",
        (x1 + 15, y1 + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "thumbs_down -> Cancel",
        (x1 + 15, y1 + 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 120, 255),
        2
    )