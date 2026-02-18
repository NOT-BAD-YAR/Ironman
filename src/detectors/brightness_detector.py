import cv2
import numpy as np
import time

class BrightnessDetector:
    def __init__(self, threshold=50, consecutive_frames=5):
        """
        Initialize Brightness detector.
        threshold: Average pixel intensity below which is considered "dark" (streetlight off).
        consecutive_frames: Number of frames the condition must persist to trigger detection.
        """
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.frame_counter = 0
        self.last_state = 'normal' # 'normal' or 'dark'

    def detect_streetlight_issue(self, frame_crop):
        """
        Analyze frame crop for low brightness indicating streetlight failure.
        Returns dict {'issue_type': 'streetlight_off', 'confidence': score} or None.
        """
        if frame_crop is None or frame_crop.size == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        detection = None
        
        if avg_brightness < self.threshold:
            self.frame_counter += 1
        else:
            self.frame_counter = 0
            
        # Check if condition persists
        if self.frame_counter >= self.consecutive_frames:
            detection = {
                'issue_type': 'streetlight_off',
                'confidence': 1.0 - (avg_brightness / 255.0), # Higher confidence for lower brightness
                'avg_brightness': avg_brightness
            }
            # Optional: Reset counter to avoid spamming? 
            # Or keeps reporting as long as it's dark.
            # Let's keep reporting, main loop can handle deduplication.
            
        return detection
