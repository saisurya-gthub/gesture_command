import time
import pyautogui
import pygetwindow as gw

pyautogui.FAILSAFE = True
ACTION_DELAY = 0.05

base_mode = "GLOBAL"
current_mode = "GLOBAL"
previous_mode = "GLOBAL"

pending_action = None
pending_target_text = ""
status_message = "System ready"


def get_active_window_title():
    """Return active window title."""
    try:
        window = gw.getActiveWindow()
        if window is None:
            return ""
        return window.title or ""
    except Exception:
        return ""


def detect_base_mode():
    """Detect mode from active application."""
    full_title = get_active_window_title()
    title_lower = full_title.lower()

    if any(word in title_lower for word in ["powerpoint", "powerpnt"]):
        return "PRESENTATION", title_lower, full_title

    if any(word in title_lower for word in ["chrome", "edge", "firefox", "brave"]):
        return "BROWSER", title_lower, full_title

    return "GLOBAL", title_lower, full_title


def sync_modes():
    """Keep base_mode updated from active app."""
    global base_mode, current_mode

    detected_mode, title_lower, full_title = detect_base_mode()
    base_mode = detected_mode

    if current_mode != "CONFIRMATION":
        current_mode = base_mode

    return title_lower, full_title


def get_status():
    """Return mode and action status for overlay."""
    _, full_title = sync_modes()

    return {
        "mode": current_mode,
        "base_mode": base_mode,
        "title": full_title,
        "pending_action": pending_action,
        "pending_target_text": pending_target_text,
        "status_message": status_message,
    }


def action_copy():
    """Copy selected content."""
    global status_message
    status_message = "Action: COPY"
    print(status_message)
    pyautogui.hotkey("ctrl", "c")
    time.sleep(ACTION_DELAY)


def action_paste():
    """Paste content."""
    global status_message
    status_message = "Action: PASTE"
    print(status_message)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(ACTION_DELAY)


def action_cut():
    """Cut selected content."""
    global status_message
    status_message = "Action: CUT"
    print(status_message)
    pyautogui.hotkey("ctrl", "x")
    time.sleep(ACTION_DELAY)


def action_delete():
    """Delete selected content."""
    global status_message
    status_message = "Action: DELETE"
    print(status_message)
    pyautogui.press("delete")
    time.sleep(ACTION_DELAY)


def action_mute():
    """Toggle mute."""
    global status_message
    status_message = "Action: MUTE"
    print(status_message)
    pyautogui.press("volumemute")
    time.sleep(ACTION_DELAY)


def action_next_tab():
    """Switch to next browser tab."""
    global status_message
    status_message = "Action: NEXT TAB"
    print(status_message)
    pyautogui.hotkey("ctrl", "tab")
    time.sleep(ACTION_DELAY)


def action_previous_tab():
    """Switch to previous browser tab."""
    global status_message
    status_message = "Action: PREVIOUS TAB"
    print(status_message)
    pyautogui.hotkey("ctrl", "shift", "tab")
    time.sleep(ACTION_DELAY)


def action_open_new_tab():
    """Open new browser tab."""
    global status_message
    status_message = "Action: OPEN NEW TAB"
    print(status_message)
    pyautogui.hotkey("ctrl", "t")
    time.sleep(ACTION_DELAY)


def action_close_tab():
    """Close current browser tab."""
    global status_message
    status_message = "Action: CLOSE TAB"
    print(status_message)
    pyautogui.hotkey("ctrl", "w")
    time.sleep(ACTION_DELAY)


def action_play_pause_youtube():
    """Toggle YouTube play/pause."""
    global status_message
    status_message = "Action: PLAY / PAUSE"
    print(status_message)
    pyautogui.press("k")   # better for YouTube than space
    time.sleep(ACTION_DELAY)


def action_start_slideshow():
    """Start slideshow."""
    global status_message
    status_message = "Action: START SLIDESHOW"
    print(status_message)
    pyautogui.press("f5")
    time.sleep(ACTION_DELAY)


def action_end_slideshow():
    """End slideshow."""
    global status_message
    status_message = "Action: END SLIDESHOW"
    print(status_message)
    pyautogui.press("esc")
    time.sleep(ACTION_DELAY)


def action_next_slide():
    """Go to next slide."""
    global status_message
    status_message = "Action: NEXT SLIDE"
    print(status_message)
    pyautogui.press("right")
    time.sleep(ACTION_DELAY)


def action_previous_slide():
    """Go to previous slide."""
    global status_message
    status_message = "Action: PREVIOUS SLIDE"
    print(status_message)
    pyautogui.press("left")
    time.sleep(ACTION_DELAY)


def switch_to_confirmation(action_name, active_mode, full_title=""):
    """Enter confirmation mode for sensitive actions."""
    global current_mode, previous_mode
    global pending_action, pending_target_text, status_message

    previous_mode = active_mode
    current_mode = "CONFIRMATION"
    pending_action = action_name

    if full_title.strip():
        pending_target_text = f"{action_name} selected in: {full_title}"
    else:
        pending_target_text = f"{action_name} selected"

    status_message = f"{action_name} selected. Show thumbs_up to proceed or thumbs_down to cancel."
    print(status_message)


def execute_pending_action():
    """Run stored confirmed action."""
    if pending_action == "DELETE":
        action_delete()


def complete_confirmation(proceeded):
    """Finish confirmation and return to previous mode."""
    global current_mode, previous_mode
    global pending_action, pending_target_text, status_message

    action_name = pending_action if pending_action else "ACTION"

    if proceeded:
        execute_pending_action()
        status_message = f"{action_name} confirmed"
    else:
        status_message = f"{action_name} cancelled"
        print(status_message)

    pending_action = None
    pending_target_text = ""
    current_mode = previous_mode


def handle_confirmation_mode(gesture_label):
    """Handle confirmation gestures only."""
    global status_message

    if gesture_label == "thumbs_up":
        complete_confirmation(True)
        return {"handled": True, "mode": "CONFIRMATION", "gesture": gesture_label}

    if gesture_label == "thumbs_down":
        complete_confirmation(False)
        return {"handled": True, "mode": "CONFIRMATION", "gesture": gesture_label}

    status_message = "Confirmation mode: thumbs_up = proceed, thumbs_down = cancel"
    print(status_message)
    return {"handled": False, "mode": "CONFIRMATION", "gesture": gesture_label}


def perform_action(gesture_label):
    """Map ML gesture labels to actions."""
    global status_message, current_mode

    title_lower, full_title = sync_modes()

    if current_mode == "CONFIRMATION":
        return handle_confirmation_mode(gesture_label)

    # ---------------- BROWSER MODE ----------------
    if current_mode == "BROWSER":
        is_youtube = "youtube" in title_lower

        # context-sensitive gesture inside browser
        if gesture_label == "v_sign":
            if is_youtube:
                action_play_pause_youtube()
                return {"handled": True, "mode": "BROWSER:YOUTUBE", "gesture": gesture_label}
            else:
                action_cut()
                return {"handled": True, "mode": "BROWSER:OTHER", "gesture": gesture_label}

        # general browser gestures
        if gesture_label == "next":
            action_next_tab()
            return {"handled": True, "mode": "BROWSER", "gesture": gesture_label}

        if gesture_label == "previous":
            action_previous_tab()
            return {"handled": True, "mode": "BROWSER", "gesture": gesture_label}

        if gesture_label == "ok_sign":
            action_open_new_tab()
            return {"handled": True, "mode": "BROWSER", "gesture": gesture_label}

        if gesture_label == "open_palm":
            action_close_tab()
            return {"handled": True, "mode": "BROWSER", "gesture": gesture_label}

        status_message = f"No browser action mapped for gesture: {gesture_label}"
        print(status_message)
        return {"handled": False, "mode": "BROWSER", "gesture": gesture_label}

    # ---------------- PRESENTATION MODE ----------------
    if current_mode == "PRESENTATION":
        if gesture_label == "next":
            action_next_slide()
            return {"handled": True, "mode": "PRESENTATION", "gesture": gesture_label}

        if gesture_label == "previous":
            action_previous_slide()
            return {"handled": True, "mode": "PRESENTATION", "gesture": gesture_label}

        if gesture_label == "ok_sign":
            action_start_slideshow()
            return {"handled": True, "mode": "PRESENTATION", "gesture": gesture_label}

        if gesture_label == "open_palm":
            action_end_slideshow()
            return {"handled": True, "mode": "PRESENTATION", "gesture": gesture_label}

        status_message = f"No presentation action mapped for gesture: {gesture_label}"
        print(status_message)
        return {"handled": False, "mode": "PRESENTATION", "gesture": gesture_label}

    # ---------------- GLOBAL MODE ----------------
    if gesture_label == "thumbs_up":
        action_copy()
        return {"handled": True, "mode": "GLOBAL", "gesture": gesture_label}

    if gesture_label == "thumbs_down":
        action_paste()
        return {"handled": True, "mode": "GLOBAL", "gesture": gesture_label}

    if gesture_label == "v_sign":
        action_cut()
        return {"handled": True, "mode": "GLOBAL", "gesture": gesture_label}

    if gesture_label == "wrapped_thumb":
        action_mute()
        return {"handled": True, "mode": "GLOBAL", "gesture": gesture_label}

    if gesture_label == "fist":
        switch_to_confirmation("DELETE", current_mode, full_title)
        return {"handled": True, "mode": "CONFIRMATION", "gesture": gesture_label}

    if gesture_label == "next":
        status_message = "Global next: temporary none"
        print(status_message)
        return {"handled": False, "mode": "GLOBAL", "gesture": gesture_label}

    if gesture_label == "previous":
        status_message = "Global previous: temporary none"
        print(status_message)
        return {"handled": False, "mode": "GLOBAL", "gesture": gesture_label}

    status_message = f"No action mapped for gesture: {gesture_label}"
    print(status_message)
    return {"handled": False, "mode": current_mode, "gesture": gesture_label}