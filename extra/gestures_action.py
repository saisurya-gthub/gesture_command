import time
import threading
import pyautogui
import pygetwindow as gw
import tkinter as tk
from tkinter import messagebox

pyautogui.FAILSAFE = True
ACTION_DELAY = 0.05  # seconds

# --------------------------------------------------
# Mode / state variables
# --------------------------------------------------
base_mode = "GLOBAL"
current_mode = "GLOBAL"
previous_mode = "GLOBAL"

pending_action = None
pending_target_text = ""
status_message = "System ready"


# --------------------------------------------------
# Helper: message box
# --------------------------------------------------
def show_message_box(title, message):
    """
    Shows a popup message in a separate thread so the
    main camera loop can continue running.
    """
    def _show():
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo(title, message)
        root.destroy()

    threading.Thread(target=_show, daemon=True).start()


# --------------------------------------------------
# Active window / mode detection
# --------------------------------------------------
def get_active_window_title():
    """
    Returns the active window title.
    """
    try:
        window = gw.getActiveWindow()
        if window is None:
            return ""
        return window.title or ""
    except Exception:
        return ""


def detect_base_mode():
    """
    Detects the base mode from the currently active application.

    Returns:
        detected_mode, title_lower, full_title
    """
    full_title = get_active_window_title()
    title_lower = full_title.lower()

    if any(word in title_lower for word in ["powerpoint", "powerpnt"]):
        return "PRESENTATION", title_lower, full_title

    if any(word in title_lower for word in ["chrome", "edge", "firefox", "brave"]):
        return "BROWSER", title_lower, full_title

    return "GLOBAL", title_lower, full_title


def sync_modes():
    """
    Keeps base_mode updated from active app.
    If we are not inside confirmation mode, current_mode follows base_mode.
    """
    global base_mode, current_mode

    detected_mode, title_lower, full_title = detect_base_mode()
    base_mode = detected_mode

    if current_mode != "CONFIRMATION":
        current_mode = base_mode

    return title_lower, full_title


def get_status():
    """
    Returns current mode/status information for overlay display.
    """
    title_lower, full_title = sync_modes()

    return {
        "mode": current_mode,
        "base_mode": base_mode,
        "title": full_title,
        "pending_action": pending_action,
        "status_message": status_message,
    }


# --------------------------------------------------
# Action functions
# --------------------------------------------------
def action_copy():
    global status_message
    status_message = "Action: COPY"
    print(status_message)
    pyautogui.hotkey("ctrl", "c")
    time.sleep(ACTION_DELAY)


def action_paste():
    global status_message
    status_message = "Action: PASTE"
    print(status_message)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(ACTION_DELAY)


def action_cut():
    global status_message
    status_message = "Action: CUT"
    print(status_message)
    pyautogui.hotkey("ctrl", "x")
    time.sleep(ACTION_DELAY)


def action_delete():
    global status_message
    status_message = "Action: DELETE"
    print(status_message)
    pyautogui.press("delete")
    time.sleep(ACTION_DELAY)


def action_mute():
    global status_message
    status_message = "Action: MUTE"
    print(status_message)
    pyautogui.press("volumemute")
    time.sleep(ACTION_DELAY)


def action_next_tab():
    global status_message
    status_message = "Action: NEXT TAB"
    print(status_message)
    pyautogui.hotkey("ctrl", "tab")
    time.sleep(ACTION_DELAY)


def action_previous_tab():
    global status_message
    status_message = "Action: PREVIOUS TAB"
    print(status_message)
    pyautogui.hotkey("ctrl", "shift", "tab")
    time.sleep(ACTION_DELAY)


def action_open_new_tab():
    global status_message
    status_message = "Action: OPEN NEW TAB"
    print(status_message)
    pyautogui.hotkey("ctrl", "t")
    time.sleep(ACTION_DELAY)


def action_close_tab():
    global status_message
    status_message = "Action: CLOSE TAB"
    print(status_message)
    pyautogui.hotkey("ctrl", "w")
    time.sleep(ACTION_DELAY)


def action_play_pause():
    global status_message
    status_message = "Action: PLAY / PAUSE"
    print(status_message)
    pyautogui.press("space")
    time.sleep(ACTION_DELAY)


def action_start_slideshow():
    global status_message
    status_message = "Action: START SLIDESHOW"
    print(status_message)
    pyautogui.press("f5")
    time.sleep(ACTION_DELAY)


def action_end_slideshow():
    global status_message
    status_message = "Action: END SLIDESHOW"
    print(status_message)
    pyautogui.press("esc")
    time.sleep(ACTION_DELAY)


def action_next_slide():
    global status_message
    status_message = "Action: NEXT SLIDE"
    print(status_message)
    pyautogui.press("right")
    time.sleep(ACTION_DELAY)


def action_previous_slide():
    global status_message
    status_message = "Action: PREVIOUS SLIDE"
    print(status_message)
    pyautogui.press("left")
    time.sleep(ACTION_DELAY)


# --------------------------------------------------
# Confirmation mode helpers
# --------------------------------------------------
def switch_to_confirmation(action_name, active_mode, full_title=""):
    """
    Switches current_mode to CONFIRMATION and stores action details.
    """
    global current_mode, previous_mode
    global pending_action, pending_target_text, status_message

    previous_mode = active_mode
    current_mode = "CONFIRMATION"
    pending_action = action_name

    if full_title.strip():
        pending_target_text = f"{action_name} selected in: {full_title}"
    else:
        pending_target_text = f"{action_name} selected"

    status_message = f"{action_name} selected. Waiting for confirmation."
    print(status_message)

    show_message_box(
        "Confirmation Required",
        f"{pending_target_text}\n\n"
        f"Show thumbs_up to proceed.\n"
        f"Show thumbs_down to cancel."
    )


def execute_pending_action():
    """
    Executes the currently pending action.
    """
    global pending_action

    if pending_action == "DELETE":
        action_delete()


def complete_confirmation(proceeded):
    """
    Completes confirmation and returns current_mode
    back to the previous mode.
    """
    global current_mode, previous_mode
    global pending_action, pending_target_text, status_message

    action_name = pending_action if pending_action else "Action"

    if proceeded:
        execute_pending_action()
        show_message_box("Confirmed", f"{action_name} executed.")
        status_message = f"{action_name} confirmed"
    else:
        show_message_box("Cancelled", f"{action_name} cancelled.")
        status_message = f"{action_name} cancelled"

    pending_action = None
    pending_target_text = ""
    current_mode = previous_mode


def handle_confirmation_mode(gesture_label):
    """
    Handles gestures only when current_mode is CONFIRMATION.
    """
    global status_message

    if gesture_label == "thumbs_up":
        complete_confirmation(True)
        return {"handled": True, "mode": "CONFIRMATION", "gesture": gesture_label}

    elif gesture_label == "thumbs_down":
        complete_confirmation(False)
        return {"handled": True, "mode": "CONFIRMATION", "gesture": gesture_label}

    else:
        status_message = "Confirmation mode: show thumbs_up to proceed or thumbs_down to cancel"
        print(status_message)
        return {"handled": False, "mode": "CONFIRMATION", "gesture": gesture_label}


# --------------------------------------------------
# Main gesture router
# --------------------------------------------------
def perform_action(gesture_label):
    """
    Mode logic:
    - base_mode is detected from the active application.
    - current_mode is the mode currently handling gestures.
    - global mode is the base support layer.
    - confirmation_mode is a general support mode used for
      2-step verification actions.
    - when a verification action is selected, current_mode
      changes to CONFIRMATION.
    - in confirmation_mode:
        thumbs_up   -> proceed
        thumbs_down -> cancel
    - after confirmation completes, current_mode returns
      to the previous/base mode.
    """
    global status_message, current_mode, base_mode

    title_lower, full_title = sync_modes()

    # --------------------------------------------------
    # 1. Confirmation mode
    # --------------------------------------------------
    if current_mode == "CONFIRMATION":
        return handle_confirmation_mode(gesture_label)

    # --------------------------------------------------
    # 2. Mode-specific actions
    # --------------------------------------------------
    if current_mode == "BROWSER":
        # YouTube-specific browser action
        if "youtube" in title_lower and gesture_label == "fist":
            action_play_pause()
            return {"handled": True, "mode": "BROWSER:YOUTUBE", "gesture": gesture_label}

        if gesture_label == "next":
            action_next_tab()
            return {"handled": True, "mode": "BROWSER", "gesture": gesture_label}

        elif gesture_label == "previous":
            action_previous_tab()
            return {"handled": True, "mode": "BROWSER", "gesture": gesture_label}

        elif gesture_label == "ok_sign":
            action_open_new_tab()
            return {"handled": True, "mode": "BROWSER", "gesture": gesture_label}

        elif gesture_label == "open_palm":
            action_close_tab()
            return {"handled": True, "mode": "BROWSER", "gesture": gesture_label}

    elif current_mode == "PRESENTATION":
        if gesture_label == "next":
            action_next_slide()
            return {"handled": True, "mode": "PRESENTATION", "gesture": gesture_label}

        elif gesture_label == "previous":
            action_previous_slide()
            return {"handled": True, "mode": "PRESENTATION", "gesture": gesture_label}

        elif gesture_label == "ok_sign":
            action_start_slideshow()
            return {"handled": True, "mode": "PRESENTATION", "gesture": gesture_label}

        elif gesture_label == "open_palm":
            action_end_slideshow()
            return {"handled": True, "mode": "PRESENTATION", "gesture": gesture_label}

    # --------------------------------------------------
    # 3. Global actions
    # --------------------------------------------------
    if gesture_label == "thumbs_up":
        action_copy()
        return {"handled": True, "mode": "GLOBAL", "gesture": gesture_label}

    elif gesture_label == "thumbs_down":
        action_paste()
        return {"handled": True, "mode": "GLOBAL", "gesture": gesture_label}

    elif gesture_label in ["v_sign", "v-sign"]:
        action_cut()
        return {"handled": True, "mode": "GLOBAL", "gesture": gesture_label}

    elif gesture_label == "wrapped_thumb":
        action_mute()
        return {"handled": True, "mode": "GLOBAL", "gesture": gesture_label}

    elif gesture_label == "fist":
        switch_to_confirmation("DELETE", current_mode, full_title)
        return {"handled": True, "mode": "CONFIRMATION", "gesture": gesture_label}

    elif gesture_label == "next":
        status_message = "Global next: temporary none"
        print(status_message)
        return {"handled": False, "mode": "GLOBAL", "gesture": gesture_label}

    elif gesture_label == "previous":
        status_message = "Global previous: temporary none"
        print(status_message)
        return {"handled": False, "mode": "GLOBAL", "gesture": gesture_label}

    status_message = f"No action mapped for gesture: {gesture_label}"
    print(status_message)
    return {"handled": False, "mode": current_mode, "gesture": gesture_label}