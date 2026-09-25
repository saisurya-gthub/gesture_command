# Gesture Based System Control

## Overview

**Gesture Based System Control** is a webcam-based application that allows users to control computer operations using **hand gestures** instead of traditional input devices such as a mouse and keyboard.

The system captures hand movements through a webcam, detects **21 hand landmarks**, and interprets them using two complementary approaches:

* **Rule-Based Gestures** – used for continuous and precise actions such as cursor movement, clicking, select and drag.
* **Static ML Gestures** – used for intentional commands such as copy, paste, cut, delete, mute, and application-specific controls.

The application also provides different **control modes**, allowing the same gesture to perform different actions depending on the active application or mode.

---

## Key Functionalities

### 1. Rule-Based Hand Control

Rule-based gestures are detected using the relative positions and distances between hand landmarks.

They are mainly used for actions that require continuous interaction.

**Examples:**

* Move the cursor using the index and middle fingers.
* **Index finger + thumb** → Left click.
* **Middle finger + thumb** → Right click.
* **Double Tap(Index finger + thumb)** → Double click.
* **Hold(Index finger + thumb) & Drag** → select and drag.

The rule-based controller has priority during normal interaction because these actions need to respond continuously and quickly.

---

### 2. Static Gesture Recognition

Static gestures are recognized using a trained **machine learning model**.

A gesture is first held in a stable position. After the required stability condition is satisfied, the ML model predicts the gesture and triggers the corresponding command.

Examples of static gestures include:

| Static Gesture    | Example Action           |
| --------------    | ------------------------ |
| 👍 Thumbs Up      | Copy                     |
| 👎 Thumbs Down    | Paste                    |
| ✌ V Sign         | Cut                      |
| 👊 Fist           | Delete with confirmation |
| ✊ Wrapped Thumb  | Mute                     |

The system uses confidence and stability checks to reduce accidental commands.

After a static command is executed, the gesture must be released before another prediction can be triggered. This **release-to-rearm mechanism** prevents the same gesture from repeatedly executing an action while it is being held.

---

## Control Modes

The application supports multiple modes. A mode determines how recognized gestures are interpreted.

### Global Mode

Global Mode provides basic system-wide controls that can be used regardless of the active application.

For example:

* 👍 Thumbs Up → Copy
* 👎 Thumbs Down → Paste
* ✌ V Sign → Cut
* 👊 Fist → Delete
* ✊ Wrapped Thumb → Mute

These commands provide a common set of controls across applications.

---

### Browser Mode

Browser Mode provides gestures specifically for web browsers.

For example:

* **👍 Thumbs Up** → Copy
* **👎 Thumbs Down** → Paste
* **✌️ V Sign** → Cut
* **👉 Next gesture** → Next browser tab
* **👈 Previous gesture** → Previous browser tab
* **👌 New-tab gesture** → Open a new tab
* **✋ Close-tab gesture** → Close the current tab

The system also checks the active browser window before performing certain application-specific actions.

For example, the **V Sign** can be interpreted as a YouTube play/pause command when a supported browser is active and the current page is YouTube.

---

### PowerPoint Mode

PowerPoint Mode maps gestures to presentation controls.

For example:

* **👉 Next gesture** → Next slide
* **👈 Previous gesture** → Previous slide
* **👌 Ok gesture** → Open slideshow
* **✋ Close gesture** → Close slideshow

This allows presentations to be controlled without physically using the keyboard or mouse.

---

### System Control Mode

System Control Mode focuses on general computer operations such as:

* Copy
* Paste
* Cut
* Delete
* Mute
* Mouse interaction

The same physical gesture can therefore have a different meaning depending on the selected mode.

---

## How Gesture Actions Change Between Modes

The important feature of the application is that **a gesture does not always represent one fixed command**.

The gesture recognition layer identifies the gesture first, while the active mode determines what action should be performed.

For example:

| Gesture           | Global Mode | Browser Mode              | PowerPoint Mode |
| --------------    | ----------- | ------------------------- | --------------- |
| 👍 Thumbs Up      | Copy        | Copy                      | Copy            |
| 👎 Thumbs Down    | Paste       | Paste                     | Paste           |
| ✌️ V Sign         | Cut         | Cut / YouTube Play-Pause* | Cut             |
| 👉 Next           | —           | Next Tab                  | Next Slide      |
| 👈 Previous       | —           | Previous Tab              | Previous Slide  |
| 👌 Ok sign        | —           | open a new Tab            | open slideshow  |
| ✋ open palm      | —           | closes Tab                | close slideshow |

*The YouTube action is performed only when the active browser/page satisfies the application's browser-specific condition.

This separation between **gesture recognition** and **action mapping** makes the system flexible and allows new modes or actions to be added without changing the basic hand-detection system.

---

## Application Workflow

The application follows a simple processing pipeline:

**Webcam → Hand Detection → Landmark Extraction → Gesture Recognition → Mode Selection → Action Execution**

1. The webcam continuously captures frames.
2. **MediaPipe Hands** detects the hand and extracts its 21 landmarks.
3. Landmark positions are processed into useful geometric features.
4. The rule-based controller checks gestures required for continuous interaction.
5. Stable static gestures are passed to the ML prediction system.
6. The active mode determines the meaning of the recognized gesture.
7. The corresponding computer action is executed using automation libraries.
8. The interface displays information such as FPS, current gesture, stable gesture, confidence, mode, and action status.

---

## Technology Stack

| Technology                     | Purpose                                                      |
| ------------------------------ | ------------------------------------------------------------ |
| **Python**                     | Core application and control logic                           |
| **OpenCV**                     | Webcam access and real-time frame processing                 |
| **MediaPipe Hands**            | Hand detection and 21-landmark tracking                      |
| **NumPy**                      | Landmark processing and numerical calculations               |
| **Scikit-learn**               | Training and prediction of static hand gestures              |
| **PyAutoGUI**                  | Mouse, keyboard, and system-level automation                 |
| **PyGetWindow / Pywinauto**    | Detecting and interacting with the active application window |
| **SVM Machine Learning Model** | Static gesture classification                                |

---

## System Architecture

The application is organized into separate components so that detection, recognition, and action execution remain independent.

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hands
   ↓
21 Hand Landmarks
   ↓
Feature Processing
   ↓
 ┌───────────────────────┐
 │                       │
Rule-Based Controller   ML Predictor
 │                       │
 │                 Stable Static Gesture
 │                       │
 └───────────┬───────────┘
             ↓
        Active Mode
             ↓
      Action Mapping
             ↓
   PyAutoGUI / Window Control
             ↓
      Computer Action
```

### Why Two Recognition Approaches?

The project uses both rule-based and ML-based recognition because different interactions have different requirements.

* **Rule-based recognition** is suitable for fast, continuous actions such as cursor movement, clicking, dragging, and scrolling.
* **ML-based recognition** is suitable for recognizing predefined static hand poses and converting them into intentional commands.
* Combining both approaches provides better control while avoiding the need to use a large number of gestures for basic mouse operations.

---

## Design Principle

The project follows the principle of **"fewer gestures, more actions."**

Instead of requiring a separate gesture for every computer operation, the system uses:

**Gesture + Mode + Active Application → Action**

This allows a relatively small set of recognizable hand gestures to control a wider range of computer operations.
