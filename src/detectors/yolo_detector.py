from ultralytics import YOLO
import cv2
import numpy as np

class YoloDetector:
    def __init__(self, model_path, conf_threshold=0.4):
        """
        Initialize YOLO detector with model weights.
        Supports both PyTorch (.pt) and ONNX (.onnx) formats.
        """
        # Specify task='detect' to avoid warning for ONNX models
        self.model = YOLO(model_path, task='detect')
        self.conf_threshold = conf_threshold
        # Allowed classes from user description
        # Note: Need to verify class IDs for 'pothole' and 'garbage' in the custom model.
        # Assuming model is trained on these classes. 
        # For now, we'll return all detections and let the caller or model handle filtering if class names are available.

    def detect(self, frame_crop):
        """
        Run YOLO detection on a frame crop.
        Returns a list of dicts: {'type': class_name, 'confidence': score, 'bbox': [x1, y1, x2, y2]}
        """
        results = self.model(frame_crop, verbose=False) # run inference
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                # Filter for pothole/garbage if implicitly required, 
                # but typically custom models only have relevant classes.
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    'type': cls_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })
                
        return detections
