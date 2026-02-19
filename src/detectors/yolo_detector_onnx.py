"""
ONNX-based YOLO Detector for Raspberry Pi 5
Optimized version using ONNX Runtime for better performance
"""

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  onnxruntime not installed. Install with: pip install onnxruntime")


class YoloDetectorONNX:
    """
    YOLO detector using ONNX Runtime (optimized for Raspberry Pi)
    """
    
    def __init__(self, model_path, conf_threshold=0.4):
        """
        Initialize ONNX YOLO detector.
        
        Args:
            model_path: Path to .onnx model file
            conf_threshold: Confidence threshold (0.0-1.0)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
        
        # Create ONNX Runtime session
        # Use CPU provider for Raspberry Pi (no GPU)
        providers = ['CPUExecutionProvider']
        
        # For better performance on Pi, you can use:
        # providers = ['CPUExecutionProvider']  # Standard CPU
        # Or if you have optimizations:
        # providers = [('CPUExecutionProvider', {'enable_mem_arena': False})]
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape (usually [1, 3, 640, 640] for YOLO)
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2] if len(self.input_shape) > 2 else 640
        
        self.conf_threshold = conf_threshold
        
        # Class names (you may need to adjust based on your model)
        # These are typical YOLO class names - update based on your model
        self.class_names = {
            0: 'drainage',
            1: 'garbage', 
            2: 'pothole',
            3: 'streetlight',
            4: 'waterleak'
        }
        
        print(f"✅ ONNX model loaded: {model_path}")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Image size: {self.img_size}x{self.img_size}")
    
    def preprocess(self, image):
        """
        Preprocess image for YOLO inference.
        
        Args:
            image: Input image (BGR format from OpenCV)
        
        Returns:
            Preprocessed image array ready for inference
        """
        # Resize to model input size
        img_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Transpose from HWC to CHW format
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension: [1, 3, 640, 640]
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch
    
    def postprocess(self, outputs, original_shape, conf_threshold=0.4):
        """
        Postprocess YOLO outputs to get detections.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (h, w)
            conf_threshold: Confidence threshold
        
        Returns:
            List of detections: [{'type': class_name, 'confidence': score, 'bbox': [x1, y1, x2, y2]}]
        """
        detections = []
        
        # YOLO output is usually [1, num_detections, 85] or similar
        # Format: [x_center, y_center, width, height, conf, class_scores...]
        output = outputs[0]  # Remove batch dimension
        
        orig_h, orig_w = original_shape[:2]
        
        # Scale factors
        scale_x = orig_w / self.img_size
        scale_y = orig_h / self.img_size
        
        for detection in output:
            # Extract box coordinates (normalized)
            x_center, y_center, width, height = detection[:4]
            conf = detection[4]
            
            # Get class scores (rest of the array)
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Combined confidence
            final_conf = conf * class_conf
            
            if final_conf < conf_threshold:
                continue
            
            # Convert from normalized center/width/height to pixel coordinates
            x_center_pixel = x_center * orig_w
            y_center_pixel = y_center * orig_h
            width_pixel = width * orig_w
            height_pixel = height * orig_h
            
            # Convert to x1, y1, x2, y2 format
            x1 = int(x_center_pixel - width_pixel / 2)
            y1 = int(y_center_pixel - height_pixel / 2)
            x2 = int(x_center_pixel + width_pixel / 2)
            y2 = int(y_center_pixel + height_pixel / 2)
            
            # Get class name
            class_name = self.class_names.get(class_id, f'class_{class_id}')
            
            detections.append({
                'type': class_name,
                'confidence': float(final_conf),
                'bbox': [x1, y1, x2, y2]
            })
        
        return detections
    
    def detect(self, frame_crop):
        """
        Run YOLO detection on a frame crop.
        
        Args:
            frame_crop: Image crop (BGR format from OpenCV)
        
        Returns:
            List of detections: [{'type': class_name, 'confidence': score, 'bbox': [x1, y1, x2, y2]}]
        """
        # Preprocess
        input_data = self.preprocess(frame_crop)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # Postprocess
        detections = self.postprocess(outputs, frame_crop.shape, self.conf_threshold)
        
        return detections


