import cv2
import time
import os
import sys
import numpy as np

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.firebase_client import get_camera_config, create_incident, upload_incident_image
from src.roi_utils import get_roi_crop, is_inside_roi
from src.detectors.yolo_detector import YoloDetector
from src.detectors.brightness_detector import BrightnessDetector

import random
import string
import json
import datetime

def main():
    # --- Configuration ---
    CAMERA_ID = "f47QoL9zBWtzs23FBjfo" # ID from user screenshot
    VIDEO_SOURCE = r"O:\CivicHeroH\src\samples\test.mp4"
    YOLO_MODEL_PATH = r"O:\CivicHeroH\ultra.pt"
    
    print(f"Starting CivicHeroH Edge for Camera: {CAMERA_ID}")
    
    # 1. Load Configuration
    camera_config = get_camera_config(CAMERA_ID)
    if not camera_config:
        print(f"Failed to load init config for {CAMERA_ID}. Exiting.")
        return
    
    rois = camera_config.get('rois', [])
    print(f"Loaded {len(rois)} ROIs from Firestore.")
    if len(rois) > 0:
        print(f"First ROI type: {rois[0].get('type')}")

    # 2. Initialize Detectors
    try:
        yolo_detector = YoloDetector(YOLO_MODEL_PATH)
        brightness_detector = BrightnessDetector()
        print("Detectors initialized.")
    except Exception as e:
        print(f"Error initializing detectors: {e}")
        return

    # 3. Open Video Source
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error opening video source: {VIDEO_SOURCE}")
        return
        
    # Get original FPS
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = int(1000 / fps)
    
    print(f"Video FPS: {fps:.2f}, Delay: {frame_delay}ms")

    # Processing Loop
    frame_count = 0
    # Process detection every N frames (skip logic is now for DETECTION only)
    DETECTION_INTERVAL = 5 
    
    # Load persistence for cooldowns
    COOLDOWN_FILE = "last_reported.json"
    if os.path.exists(COOLDOWN_FILE):
        try:
            with open(COOLDOWN_FILE, 'r') as f:
                last_reported = json.load(f)
        except:
            last_reported = {}
    else:
        last_reported = {} # {roi_id: timestamp}
        
    REPORT_COOLDOWN =60     #86400 # 24 Hours (86400 seconds)
    
    visual_feedback_text = ""
    visual_feedback_start = 0
    
    # Store detections to draw on intermediate frames
    active_detections = [] # List of tuples: (bbox, label, conf)
    
    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of video stream. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        frame_count += 1
        debug_frame = frame.copy()
        current_time = time.time()
        
        # --- DETECTION PHASE (Run every N frames) ---
        if frame_count % DETECTION_INTERVAL == 0:
            active_detections = [] # Clear previous detections
            
            for roi in rois:
                roi_id = roi.get('id')
                roi_type = roi.get('type')
                points = roi.get('points')
                
                # Crop & Detect
                roi_crop = get_roi_crop(frame, points)
                
                # 1. YOLO Detection
                if roi_type in ['pothole', 'garbage', 'road']:
                    detections = yolo_detector.detect(roi_crop)
                    for det in detections:
                        label = det['type']
                        conf = det['confidence']
                        bbox = det['bbox'] # local bbox
                        
                        # Convert to global for display
                        h, w = frame.shape[:2]
                        pixel_points = np.array([[int(p[0]*w), int(p[1]*h)] for p in points], np.int32)
                        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(pixel_points)
                        global_bbox = [bbox[0]+roi_x, bbox[1]+roi_y, bbox[2]+roi_x, bbox[3]+roi_y]
                        
                        # Store for drawing
                        active_detections.append((global_bbox, label, conf))
                        
                        # REPORTING LOGIC
                        matched = False
                        if roi_type == 'road' and label == 'pothole': matched = True
                        elif roi_type == label: matched = True
                        
                        if matched:
                            # Check cooldown
                            if roi_id in last_reported and (current_time - last_reported[roi_id] < REPORT_COOLDOWN):
                                continue # Skip reporting, but keep visualizing
                            
                            # Report Incident (Same logic as before)
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                            random_suffix = ''.join(random.choices(string.digits, k=4))
                            complain_id = f"CH{timestamp_str}{random_suffix}"
                            
                            image_filename = f"{complain_id}.jpg"
                            temp_image_path = os.path.join(os.path.dirname(__file__), 'samples', 'temp_captures', image_filename)
                            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                            cv2.imwrite(temp_image_path, roi_crop)
                            
                            print(f"Uploading image for {complain_id}...")
                            remote_path = f"issues/{image_filename}"
                            image_url = upload_incident_image(temp_image_path, remote_path)
                            
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)
                                
                            department_map = {'pothole': 'Road Department', 'garbage': 'Sanitation Department', 'streetlight': 'Electricity Department', 'water_leak': 'Water Department'}
                            department = department_map.get(label, 'General Department')

                            urgency = 'Low'
                            if label == 'pothole': urgency = 'High' if conf > 0.8 else 'Medium'
                            elif label == 'garbage': urgency = 'Medium'
                            
                            address = camera_config.get('locationName', camera_config.get('address', 'Unknown Location'))
                            user_id = camera_config.get('user_id', 'unknown_user') 
                            lat = camera_config.get('latitude', 0.0)
                            lng = camera_config.get('longitude', 0.0)
                            reported_date = datetime.datetime.now().isoformat()

                            incident_data = {
                                "address": address, "complain_id": complain_id, "department": department,
                                "description": f"Automated detection: {label} with {conf:.2f} confidence.",
                                "image_url": image_url or "", "issue_type": label.capitalize(), 
                                "latitude": lat, "longitude": lng, "reported_date": reported_date,
                                "status": "Reported", "urgency": urgency, "user_id": user_id,
                                "roi_id": roi_id, "camera_id": CAMERA_ID
                            }
                            
                            create_incident(incident_data)
                            last_reported[roi_id] = time.time()
                            try:
                                with open(COOLDOWN_FILE, 'w') as f: json.dump(last_reported, f)
                            except Exception as e: print(f"Error saving cooldowns: {e}")
                            
                            visual_feedback_text = f"SENT: {label} ({urgency})"
                            visual_feedback_start = time.time()
                            break 

                # 2. Streetlight Detection
                if roi_type == 'streetlight':
                    result = brightness_detector.detect_streetlight_issue(roi_crop)
                    if result:
                        # Use center of ROI for "detection box" visualization
                        h, w = frame.shape[:2]
                        pixel_points = np.array([[int(p[0]*w), int(p[1]*h)] for p in points], np.int32)
                        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(pixel_points)
                        # Draw box around whole ROI
                        active_detections.append(([roi_x, roi_y, roi_x+roi_w, roi_y+roi_h], 'Streetlight', result['confidence']))
                        
                        # Check cooldown
                        if not (roi_id in last_reported and (current_time - last_reported[roi_id] < REPORT_COOLDOWN)):
                            # Reporting Logic...
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                            random_suffix = ''.join(random.choices(string.digits, k=4))
                            complain_id = f"CH{timestamp_str}{random_suffix}"
                            
                            image_filename = f"{complain_id}.jpg"
                            temp_image_path = os.path.join(os.path.dirname(__file__), 'samples', 'temp_captures', image_filename)
                            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                            cv2.imwrite(temp_image_path, roi_crop)
                            
                            print(f"Uploading image for {complain_id}...")
                            remote_path = f"issues/{image_filename}"
                            image_url = upload_incident_image(temp_image_path, remote_path)
                            
                            if os.path.exists(temp_image_path): os.remove(temp_image_path)

                            incident_data = {
                                "address": camera_config.get('locationName', camera_config.get('address', 'Unknown Location')),
                                "complain_id": complain_id, "department": 'Electricity Department',
                                "description": "Streetlight appears to be off at night",
                                "image_url": image_url or "", "issue_type": "Streetlight",
                                "latitude": camera_config.get('latitude', 0.0), "longitude": camera_config.get('longitude', 0.0),
                                "reported_date": datetime.datetime.now().isoformat(),
                                "status": "Reported", "urgency": "Low",
                                "user_id": camera_config.get('user_id', 'unknown_user'),
                                "roi_id": roi_id, "camera_id": CAMERA_ID
                            }
                            create_incident(incident_data)
                            last_reported[roi_id] = time.time()
                            try:
                                with open(COOLDOWN_FILE, 'w') as f: json.dump(last_reported, f)
                            except Exception as e: print(f"Error saving cooldowns: {e}")
                            
                            visual_feedback_text = "SENT: Streetlight Failure"
                            visual_feedback_start = time.time()

        # --- DRAWING PHASE (Every Frame) ---
        # Draw ROIs
        for roi in rois:
            points = roi.get('points')
            h, w = frame.shape[:2]
            pixel_points = np.array([[int(p[0]*w), int(p[1]*h)] for p in points], np.int32)
            cv2.polylines(debug_frame, [pixel_points], True, (0, 255, 0), 2)
            cv2.putText(debug_frame, roi.get('id'), (pixel_points[0][0], pixel_points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Draw Active Detections (persisted from last detection frame)
        for det in active_detections:
            g_bbox, label, conf = det
            cv2.rectangle(debug_frame, (int(g_bbox[0]), int(g_bbox[1])), (int(g_bbox[2]), int(g_bbox[3])), (0, 0, 255), 2)
            cv2.putText(debug_frame, f"{label} {conf:.2f}", (int(g_bbox[0]), int(g_bbox[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw Feedback Overlay
        if current_time - visual_feedback_start < 3.0 and visual_feedback_text:
            cv2.putText(debug_frame, visual_feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('CivicHeroH Edge Debug', debug_frame)
        
        # Calculate time spent
        elapsed = (time.time() - loop_start) * 1000 # ms
        wait_ms = max(1, frame_delay - int(elapsed))
        
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
