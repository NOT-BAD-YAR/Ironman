# CivicHeroH Project Documentation

## Project Overview
**Project Folder:** `O:\CivicHeroH`

This project is an edge computing solution for detecting civic issues like potholes, garbage, and streetlight failures using camera feeds.

## Root Files

### `ultra.pt`
- **What it is:** Trained YOLO model weights.
- **Why:** YOLO needs this file to detect pothole/garbage classes.
- **Used by:** `src/detectors/yolo_detector.py`

### `serviceAccountKey.json`
- **What it is:** Firebase Admin key file (server-side access).
- **Why:** Lets Python read camera configs + write incidents to Firestore.
- **Used by:** `src/firebase_client.py`

### `reeq.txt` (Requirements)
- **What it is:** List of Python packages needed.
- **Why:** Makes installation easy on any machine.
- **Example packages:** `ultralytics`, `opencv-python`, `firebase-admin`, `numpy`
*(Note: User referred to this as `requirements.txt`)*

## Source Code (`src/`)

### `src/main.py` (Controller / Orchestrator)
The "brain" of the edge program.
- **Functionality:**
    - Loads camera + ROI configuration from Firestore.
    - Opens video source (MP4 now, RTSP later).
    - Runs the processing loop (burst sampling / low FPS).
    - Coordinates detection:
        - Sends ROIs to correct detector (YOLO or brightness).
        - If an issue is confirmed, asks Firebase client to create an incident record.

✅ **Summary:** `main.py` coordinates everything.

### `src/firebase_client.py` (DB Read/Write Helper)
Handles all Firebase interactions.
- **Functionality:**
    - Initializes Firebase Admin SDK.
    - Provides functions like:
        - `get_camera(camera_id)`
        - `get_regions(camera_id)`
        - `create_incident(data)`
        - (Optional) `upload_snapshot(image)` if using Firebase Storage.

✅ **Summary:** All Firestore/Storage code typically resides here.

### `src/roi_utils.py` (ROI Math Tools)
Converts ROI points into usable crops. Points are stored as normalized (0–1) coordinates.
- **Functionality:**
    - Converts normalized points → pixel points using frame width/height.
    - Crops a polygon area or bounding-box crop.
    - Checks if a detected object bbox center is inside ROI (optional).
    - Utility functions to simplify detectors.

✅ **Summary:** Turns ROI points into actual image regions.

## Detectors (`src/detectors/`)

### `src/detectors/yolo_detector.py` (Pothole/Garbage Detection)
- **Functionality:**
    - Loads `ultra.pt` using Ultralytics YOLO.
    - Implements `detect(frame_crop) -> detections`.
    - Filters detections by confidence threshold and allowed classes (pothole, garbage).
    - Outputs: `issue_type`, `confidence`, `bbox`.

✅ **Summary:** Runs YOLO on cropped image.

### `src/detectors/brightness_detector.py` (Streetlight "Off" Detection)
Does NOT use YOLO.
- **Functionality:**
    - Takes streetlight ROI crop.
    - Converts to grayscale.
    - Computes brightness average.
    - Compares with a threshold, neighbor ROIs, or baseline brightness.
    - Uses time confirmation (low brightness for N checks).
    - Outputs: `issue_type = streetlight_off`, `confidence/score`.

✅ **Summary:** Detects streetlight failure by brightness.

## Samples (`src/samples/`)

### `src/samples/done.mp4`
- **Purpose:** Used for simulation instead of real CCTV.
- **Benefit:** Lets you test your entire pipeline safely.
*(Note: User referred to this as `samples/test.mp4`, but `done.mp4` was found in `src/samples/`)*
"# Ironman" 
