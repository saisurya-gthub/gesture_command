import math


def distance_3d(a, b) -> float:
    """Return 3D distance between two landmarks."""
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        (a.z - b.z) ** 2
    )


def normalize_landmarks(landmarks):
    """Normalize 21 landmarks into flat 63-value feature vector."""
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    scale = distance_3d(wrist, middle_mcp)
    if scale == 0:
        scale = 1.0

    normalized = []
    for lm in landmarks:
        normalized.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale
        ])

    return normalized