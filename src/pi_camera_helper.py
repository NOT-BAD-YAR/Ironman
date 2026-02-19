"""
Raspberry Pi Camera Helper
Provides easy integration with Raspberry Pi Camera Module using picamera2
"""

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  picamera2 not available. Install with: sudo apt install python3-picamera2")


class PiCameraHelper:
    """
    Helper class for Raspberry Pi Camera Module integration
    """
    
    def __init__(self, width=1920, height=1080, framerate=30):
        """
        Initialize Raspberry Pi Camera
        
        Args:
            width: Frame width (default: 1920)
            height: Frame height (default: 1080)
            framerate: Frames per second (default: 30)
        """
        if not PICAMERA2_AVAILABLE:
            raise ImportError("picamera2 is not installed. Install with: sudo apt install python3-picamera2")
        
        self.picam2 = Picamera2()
        
        # Configure video capture
        video_config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": framerate}
        )
        self.picam2.configure(video_config)
        self.picam2.start()
        
        print(f"✅ Raspberry Pi Camera initialized: {width}x{height} @ {framerate}fps")
    
    def read(self):
        """
        Read a frame from the camera
        
        Returns:
            ret: True if frame was captured
            frame: BGR frame (OpenCV format)
        """
        try:
            # Capture frame (RGB format)
            frame_rgb = self.picam2.capture_array()
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            return True, frame_bgr
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        if hasattr(self, 'picam2'):
            self.picam2.stop()
            print("Camera released")
    
    def get(self, prop):
        """
        Get camera property (for compatibility with cv2.VideoCapture)
        
        Args:
            prop: Property constant (e.g., cv2.CAP_PROP_FPS)
        
        Returns:
            Property value
        """
        if prop == cv2.CAP_PROP_FPS:
            return 30.0  # Default framerate
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 1920.0
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 1080.0
        return 0.0
    
    def set(self, prop, value):
        """
        Set camera property (for compatibility with cv2.VideoCapture)
        Note: Pi Camera properties are set during initialization
        """
        pass
    
    def isOpened(self):
        """Check if camera is opened"""
        return PICAMERA2_AVAILABLE and hasattr(self, 'picam2')


def create_camera_source(source):
    """
    Create appropriate camera source based on input
    
    Args:
        source: Can be:
            - "pi" or "picamera": Use Raspberry Pi Camera Module
            - int: USB camera index (0, 1, etc.)
            - str: RTSP URL or file path
    
    Returns:
        Camera object (PiCameraHelper or cv2.VideoCapture)
    """
    if isinstance(source, str) and source.lower() in ["pi", "picamera", "raspberry"]:
        if PICAMERA2_AVAILABLE:
            return PiCameraHelper()
        else:
            print("⚠️  picamera2 not available, falling back to USB camera index 0")
            return cv2.VideoCapture(0)
    elif isinstance(source, int):
        # USB camera
        return cv2.VideoCapture(source)
    else:
        # RTSP stream or file path
        return cv2.VideoCapture(source)


