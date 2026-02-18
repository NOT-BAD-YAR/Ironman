import cv2
import numpy as np

def convert_normalized_to_pixel(point, frame_width, frame_height):
    """
    Convert a normalized point (0-1) to pixel coordinates.
    """
    x = int(point[0] * frame_width)
    y = int(point[1] * frame_height)
    return (x, y)

def get_roi_crop(frame, polygon_points):
    """
    Crop the frame based on a polygon ROI.
    Returns the cropped image.
    """
    # Create a mask for the polygon
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Convert points to numpy array of scaling to frame size
    h, w = frame.shape[:2]
    pixel_points = np.array([convert_normalized_to_pixel(p, w, h) for p in polygon_points], dtype=np.int32)
    
    # Fill the polygon on the mask
    cv2.fillPoly(mask, [pixel_points], 255)
    
    # Apply the mask
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Crop the bounding rect of the polygon
    x, y, w, h = cv2.boundingRect(pixel_points)
    crop = masked_image[y:y+h, x:x+w]
    
    return crop

def is_inside_roi(bbox, polygon_points, frame_width, frame_height):
    """
    Check if the center of a bounding box is inside the ROI polygon.
    bbox: [x1, y1, x2, y2]
    """
    # Calculate center of bbox
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    
    # Convert polygon points to pixel coordinates
    pixel_points = np.array([convert_normalized_to_pixel(p, frame_width, frame_height) for p in polygon_points], dtype=np.int32)
    
    # Check if point is inside polygon
    # measureDist=False returns +1 if inside, -1 if outside, 0 if on edge
    result = cv2.pointPolygonTest(pixel_points, (cx, cy), False)
    
    return result >= 0
