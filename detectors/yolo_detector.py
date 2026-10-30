"""
YOLO Vehicle Detection Module

Detects vehicles (Cars, Buses, Trucks) using Ultralytics YOLO model.
Provides temporary vehicle labels until MobileNetV2 classification is applied.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOVehicleDetector:
    """YOLO-based vehicle detector for Cars, Buses, and Trucks."""
    
    # COCO class IDs for vehicle categories
    VEHICLE_CLASSES = {
        2: 'Car',
        3: 'Car', 
        5: 'Bus',
        7: 'Bus',
        8: 'Bus',
        39: 'Car',  # taxi (treated as car)
        40: 'Truck',
        41: 'Truck',
        42: 'Truck',
        43: 'Truck',
        44: 'Truck',
        45: 'Truck',
        46: 'Truck',
        47: 'Truck',
        48: 'Truck',
        49: 'Truck',
        50: 'Truck',
        51: 'Truck',
        52: 'Truck',
        53: 'Truck',
        54: 'Truck',
        55: 'Truck',
        56: 'Truck',
        57: 'Truck',
        58: 'Truck',
        59: 'Truck',
        60: 'Truck',
        61: 'Truck',
        62: 'Truck',
        63: 'Truck',
        64: 'Truck',
        65: 'Truck',
        66: 'Truck',
        67: 'Truck',
        68: 'Truck',
        69: 'Truck',
        70: 'Truck',
        71: 'Truck',
        72: 'Truck',
        73: 'Truck',
        74: 'Truck',
        75: 'Truck',
        76: 'Truck',
        77: 'Truck',
        78: 'Truck',
        79: 'Truck',
        80: 'Truck',
        81: 'Truck',
        82: 'Truck',
        84: 'Truck',
        85: 'Truck',
        86: 'Truck',
        87: 'Truck',
        88: 'Truck',
        89: 'Truck',
        90: 'Truck',
    }
    
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.25):
        """
        Initialize the YOLO vehicle detector.
        
        Args:
            model_name: Path to YOLO model file (default: yolov8n.pt - pretrained on COCO)
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        
        # Load YOLO model - this downloads pretrained COCO weights
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        print("YOLO model loaded successfully")
    
    def detect(self, frame):
        """
        Detect vehicles in a frame.
        
        Args:
            frame: numpy array representing the image/frame
            
        Returns:
            list of dicts with keys: 'bbox', 'class_name', 'confidence', 'class_id'
        """
        if frame is None or frame.size == 0:
            return []
        
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names.get(class_id, f'class_{class_id}')
                
                # Only include vehicle classes
                if class_id in self.VEHICLE_CLASSES:
                    normalized_class = self.VEHICLE_CLASSES.get(class_id, class_name)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'class_name': normalized_class,
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'source': 'YOLO'
                    })
        
        return detections
    
    def annotate_frame(self, frame, detections):
        """
        Annotate frame with detection bounding boxes.
        
        Args:
            frame: numpy array representing the image/frame
            detections: list of detection dicts from detect()
            
        Returns:
            annotated frame with bounding boxes and labels
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated