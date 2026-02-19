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

# Try to import Pi Camera helper (optional)
try:
    from src.pi_camera_helper import create_camera_source
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False

import random
import string
import json
import datetime

def main():
    # --- Configuration ---
    CAMERA_ID = "f47QoL9zBWtzs23FBjfo" # ID from user screenshot
    
    # Video source priority:
    # 1. Use test video first (for testing)
    # 2. Fallback to webcam if video not found
    # 3. Future: Can use RTSP stream for CCTV camera
    # Examples:
    #   - Local file: "src/samples/test.mp4"
    #   - Webcam: 0 (default camera)
    #   - RTSP stream: "rtsp://username:password@ip:port/stream"
    
    # Primary: Use test video
    VIDEO_SOURCE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "samples", "test.mp4")
    FALLBACK_WEBCAM = 0  # Webcam (fallback if video not found)
    
    # YOLO model path - Using ONNX for better performance
    YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ultra.onnx")
    
    print(f"Starting CivicHeroH Edge for Camera: {CAMERA_ID}")
    
    # 1. Load Configuration (with timeout and fallback)
    print("📡 Connecting to Firebase...")
    camera_config = None
    rois = []
    
    try:
        # Try to get camera config (non-blocking with timeout)
        import threading
        
        config_result = [None]
        config_error = [None]
        
        def fetch_config():
            try:
                config_result[0] = get_camera_config(CAMERA_ID)
            except Exception as e:
                config_error[0] = e
        
        # Start config fetch in background thread
        config_thread = threading.Thread(target=fetch_config, daemon=True)
        config_thread.start()
        config_thread.join(timeout=5)  # Wait max 5 seconds
        
        if config_thread.is_alive():
            print("⚠️  Firebase connection timeout (>5s)")
            print("⚠️  Continuing without Firebase - detection will still work")
            camera_config = None
        elif config_error[0]:
            print(f"⚠️  Firebase connection error: {config_error[0]}")
            print("⚠️  Continuing without Firebase - detection will still work")
            camera_config = None
        else:
            camera_config = config_result[0]
            
    except Exception as e:
        print(f"⚠️  Error loading camera config: {e}")
        print("⚠️  Continuing without Firebase - detection will still work")
        camera_config = None
    
    if camera_config:
        rois = camera_config.get('rois', [])
        print(f"✅ Loaded {len(rois)} ROIs from Firestore.")
        if len(rois) > 0:
            print(f"   First ROI type: {rois[0].get('type')}")
    else:
        print("⚠️  No camera config loaded - detection will work on entire frame")
        print("   Note: All detections will be processed (no ROI filtering)")
        rois = []

    # 2. Initialize Detectors
    try:
        print(f"🤖 Loading ONNX model: {YOLO_MODEL_PATH}")
        yolo_detector = YoloDetector(YOLO_MODEL_PATH)
        brightness_detector = BrightnessDetector()
        print("✅ Detectors initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing detectors: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Open Video Source with fallback
    cap = None
    video_source_name = "Unknown"
    
    # Try test video first (for testing)
    try:
        print(f"📹 Attempting to open test video: {VIDEO_SOURCE}")
        if os.path.exists(VIDEO_SOURCE):
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if cap.isOpened():
                # Test if video actually works by reading a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    video_source_name = "Test Video (test.mp4)"
                    print(f"✅ Successfully opened {video_source_name}")
                    # Reset to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    print(f"⚠️  Video opened but cannot read frames")
                    cap.release()
                    cap = None
            else:
                print(f"⚠️  Cannot open video file")
                cap = None
        else:
            print(f"⚠️  Video file not found: {VIDEO_SOURCE}")
            cap = None
    except Exception as e:
        print(f"⚠️  Error opening video: {e}")
        if cap:
            cap.release()
        cap = None
    
    # Fallback to webcam if video fails
    if cap is None or not cap.isOpened():
        try:
            print(f"📹 Falling back to webcam (index {FALLBACK_WEBCAM})...")
            cap = cv2.VideoCapture(FALLBACK_WEBCAM)
            
            # Test if webcam actually works by reading a frame
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    video_source_name = f"Webcam (index {FALLBACK_WEBCAM})"
                    print(f"✅ Successfully connected to {video_source_name}")
                else:
                    print(f"⚠️  Webcam opened but cannot read frames")
                    cap.release()
                    cap = None
            else:
                print(f"⚠️  Cannot open webcam")
                cap = None
        except Exception as e:
            print(f"⚠️  Error opening webcam: {e}")
            if cap:
                cap.release()
            cap = None
    
    if not cap or not cap.isOpened():
        print("❌ Failed to open any video source. Exiting.")
        return
    
    # Support for Raspberry Pi Camera Module (future use)
    if PI_CAMERA_AVAILABLE and isinstance(VIDEO_SOURCE, str) and VIDEO_SOURCE.lower() in ["pi", "picamera", "raspberry"]:
        try:
            pi_cap = create_camera_source(VIDEO_SOURCE)
            if pi_cap.isOpened():
                if cap:
                    cap.release()
                cap = pi_cap
                video_source_name = "Raspberry Pi Camera Module"
                print(f"✅ Using {video_source_name}")
        except Exception as e:
            print(f"⚠️  Pi Camera not available: {e}")
            # Continue with existing cap
    
    if not cap or not cap.isOpened():
        print("❌ Failed to open any video source. Exiting.")
        return
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = int(1000 / fps) if fps > 0 else 33  # Default 30fps if unknown
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video Source: {video_source_name}")
    print(f"📊 Resolution: {width}x{height} @ {fps:.2f} FPS")
    print(f"⏱️  Frame delay: {frame_delay}ms")

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
            
            # Debug: Show detection status
            if frame_count == DETECTION_INTERVAL:
                print(f"🔍 Starting detection (ROIs: {len(rois)}, Interval: every {DETECTION_INTERVAL} frames)")
            
            # If no ROIs configured, process entire frame
            if len(rois) == 0:
                # Process entire frame without ROI filtering
                try:
                    detections = yolo_detector.detect(frame)
                    print(f"🔍 Frame {frame_count}: Found {len(detections)} detections (no ROIs configured)")
                    
                    for det in detections:
                        label = det['type']
                        conf = det['confidence']
                        bbox = det['bbox']
                        
                        print(f"   → {label}: {conf:.2f} confidence at {bbox}")
                        
                        # Store for drawing
                        active_detections.append((bbox, label, conf))
                        
                        # Report if confidence is high enough
                        if conf >= 0.5:  # Threshold for reporting without ROI
                            # Check cooldown (use 'full_frame' as ROI ID)
                            roi_id = 'full_frame'
                            if roi_id in last_reported and (current_time - last_reported[roi_id] < REPORT_COOLDOWN):
                                print(f"   ⏸️  Cooldown active, skipping report")
                                continue
                            
                            # Report incident
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                            random_suffix = ''.join(random.choices(string.digits, k=4))
                            complain_id = f"CH{timestamp_str}{random_suffix}"
                            
                            image_filename = f"{complain_id}.jpg"
                            temp_image_path = os.path.join(os.path.dirname(__file__), 'samples', 'temp_captures', image_filename)
                            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                            cv2.imwrite(temp_image_path, frame)  # Save full frame
                            
                            print(f"📤 Uploading image for {complain_id}...")
                            remote_path = f"issues/{image_filename}"
                            image_url = ""
                            if camera_config:  # Only upload if Firebase connected
                                image_url = upload_incident_image(temp_image_path, remote_path)
                            
                            if os.path.exists(temp_image_path):
                                os.remove(temp_image_path)
                            
                            department_map = {'pothole': 'Road Department', 'garbage': 'Sanitation Department', 'streetlight': 'Electricity Department', 'water_leak': 'Water Department'}
                            department = department_map.get(label, 'General Department')
                            
                            urgency = 'Low'
                            if label == 'pothole': urgency = 'High' if conf > 0.8 else 'Medium'
                            elif label == 'garbage': urgency = 'Medium'
                            
                            # Get location from camera config (with fallback)
                            if camera_config:
                                address = camera_config.get('locationName', camera_config.get('address', 'Unknown Location'))
                                user_id = camera_config.get('user_id', 'unknown_user')
                                lat = camera_config.get('latitude', 0.0)
                                lng = camera_config.get('longitude', 0.0)
                            else:
                                address = 'Unknown Location (Firebase not connected)'
                                user_id = 'unknown_user'
                                lat = 0.0
                                lng = 0.0
                            
                            reported_date = datetime.datetime.now().isoformat()
                            
                            incident_data = {
                                "address": address, "complain_id": complain_id, "department": department,
                                "description": f"Automated detection: {label} with {conf:.2f} confidence.",
                                "image_url": image_url or "", "issue_type": label.capitalize(), 
                                "latitude": lat, "longitude": lng, "reported_date": reported_date,
                                "status": "Reported", "urgency": urgency, "user_id": user_id,
                                "roi_id": roi_id, "camera_id": CAMERA_ID
                            }
                            
                            # Always try to create incident (Firebase client handles connection check)
                            try:
                                create_incident(incident_data)
                                print(f"✅ Incident created in Firebase: {complain_id}")
                            except Exception as e:
                                print(f"⚠️  Failed to create incident in Firebase: {e}")
                                print(f"   Incident details: {complain_id} ({label}, {conf:.2f})")
                            
                            last_reported[roi_id] = time.time()
                            try:
                                with open(COOLDOWN_FILE, 'w') as f: json.dump(last_reported, f)
                            except Exception as e: print(f"Error saving cooldowns: {e}")
                            
                            visual_feedback_text = f"DETECTED: {label} ({urgency})"
                            visual_feedback_start = time.time()
                except Exception as e:
                    print(f"❌ Detection error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Process ROIs if configured
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
                            
                            # Always try to create incident (Firebase client handles connection check)
                            try:
                                create_incident(incident_data)
                                print(f"✅ Incident created in Firebase: {complain_id}")
                            except Exception as e:
                                print(f"⚠️  Failed to create incident in Firebase: {e}")
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

                            # Get location from camera config (with fallback)
                            if camera_config:
                                address = camera_config.get('locationName', camera_config.get('address', 'Unknown Location'))
                                user_id = camera_config.get('user_id', 'unknown_user')
                                lat = camera_config.get('latitude', 0.0)
                                lng = camera_config.get('longitude', 0.0)
                            else:
                                address = 'Unknown Location (Firebase not connected)'
                                user_id = 'unknown_user'
                                lat = 0.0
                                lng = 0.0
                            
                            incident_data = {
                                "address": address,
                                "complain_id": complain_id, "department": 'Electricity Department',
                                "description": "Streetlight appears to be off at night",
                                "image_url": image_url or "", "issue_type": "Streetlight",
                                "latitude": lat, "longitude": lng,
                                "reported_date": datetime.datetime.now().isoformat(),
                                "status": "Reported", "urgency": "Low",
                                "user_id": user_id,
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
        
        # Show detection status on frame
        status_text = f"Frame: {frame_count} | Detections: {len(active_detections)}"
        if len(rois) == 0:
            status_text += " | Mode: Full Frame (No ROIs)"
        else:
            status_text += f" | ROIs: {len(rois)}"
        cv2.putText(debug_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
