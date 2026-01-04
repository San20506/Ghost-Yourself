"""YOLOv8 segmentation-based object detector."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from ...config import DetectionConfig


# COCO class names for reference
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}


@dataclass
class Detection:
    """Represents a single detected object."""
    
    id: int  # Unique ID for this detection
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None  # Segmentation mask (same size as frame)
    centroid: Tuple[int, int] = (0, 0)
    
    def __post_init__(self):
        """Calculate centroid from bounding box."""
        x1, y1, x2, y2 = self.bbox
        self.centroid = ((x1 + x2) // 2, (y1 + y2) // 2)


class ObjectDetector:
    """YOLOv8 segmentation model wrapper for object detection."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize the detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self._detection_id_counter = 0
        
        # Load YOLO model
        print(f"Loading YOLOv8 model: {config.model_name}")
        self.model = YOLO(config.model_name)
        
        # Set device
        if config.device:
            self.model.to(config.device)
        
        print(f"Model loaded on device: {self.model.device}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Run inference
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return detections
        
        result = results[0]  # First (and only) image result
        
        # Check if we have detections
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Get frame dimensions for mask resizing
        frame_h, frame_w = frame.shape[:2]
        
        # Process each detection
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            
            # Filter by target classes if specified
            if (self.config.target_classes is not None and 
                class_id not in self.config.target_classes):
                continue
            
            # Extract bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Get segmentation mask if available
            mask = None
            if result.masks is not None and i < len(result.masks):
                # Masks are at model resolution, need to resize to frame
                mask_data = result.masks[i].data.cpu().numpy()[0]
                mask = cv2.resize(mask_data, (frame_w, frame_h))
                mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Create detection object
            detection = Detection(
                id=self._detection_id_counter,
                class_id=class_id,
                class_name=COCO_CLASSES.get(class_id, f"class_{class_id}"),
                confidence=float(box.conf[0]),
                bbox=(x1, y1, x2, y2),
                mask=mask
            )
            
            detections.append(detection)
            self._detection_id_counter += 1
        
        return detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        selected_id: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        selected_color: Tuple[int, int, int] = (0, 120, 255),
        thickness: int = 2,
        show_labels: bool = True
    ) -> np.ndarray:
        """Draw detection boxes on frame.
        
        Args:
            frame: Frame to draw on
            detections: List of detections
            selected_id: ID of selected detection (different color)
            color: Default box color
            selected_color: Color for selected detection
            thickness: Line thickness
            show_labels: Whether to show class labels
            
        Returns:
            Frame with drawn detections
        """
        result = frame.copy()
        
        for det in detections:
            # Choose color based on selection
            box_color = selected_color if det.id == selected_id else color
            
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), box_color, thickness)
            
            if show_labels:
                # Draw label background
                label = f"{det.class_name} {det.confidence:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(
                    result,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 10, y1),
                    box_color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    result, label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA
                )
        
        return result
