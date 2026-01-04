"""Object selection and tracking for persistent selection across frames."""

from typing import List, Optional, Tuple

import numpy as np

from .detector import Detection
from ...common.utils import point_in_box


class ObjectTracker:
    """Handles object selection via click and tracks selection across frames."""
    
    def __init__(self):
        """Initialize the tracker."""
        self.selected_detection: Optional[Detection] = None
        self.selected_class_id: Optional[int] = None
        self.last_centroid: Optional[Tuple[int, int]] = None
        
        # Tracking settings
        self.max_centroid_distance = 150  # Max distance to match between frames
    
    def select_at_point(
        self,
        point: Tuple[int, int],
        detections: List[Detection]
    ) -> Optional[Detection]:
        """Select an object at the given point.
        
        Args:
            point: (x, y) click coordinates
            detections: Current frame detections
            
        Returns:
            Selected detection if found, None otherwise
        """
        # Find detection containing the point
        for det in detections:
            if point_in_box(point, det.bbox):
                self.selected_detection = det
                self.selected_class_id = det.class_id
                self.last_centroid = det.centroid
                return det
        
        # No detection at point - clear selection
        self.clear_selection()
        return None
    
    def track(self, detections: List[Detection]) -> Optional[Detection]:
        """Find the previously selected object in new detections.
        
        Uses a combination of class ID and centroid proximity to
        maintain consistent selection across frames.
        
        Args:
            detections: Current frame detections
            
        Returns:
            Matched detection if found, None otherwise
        """
        if self.selected_class_id is None or self.last_centroid is None:
            return None
        
        # Filter to same class
        same_class = [d for d in detections if d.class_id == self.selected_class_id]
        
        if not same_class:
            # Object gone from frame
            return None
        
        # Find closest by centroid
        best_match = None
        best_distance = float('inf')
        
        for det in same_class:
            dist = self._centroid_distance(det.centroid, self.last_centroid)
            if dist < best_distance and dist < self.max_centroid_distance:
                best_distance = dist
                best_match = det
        
        if best_match:
            self.selected_detection = best_match
            self.last_centroid = best_match.centroid
        
        return best_match
    
    def clear_selection(self):
        """Clear the current selection."""
        self.selected_detection = None
        self.selected_class_id = None
        self.last_centroid = None
    
    def has_selection(self) -> bool:
        """Check if an object is currently selected."""
        return self.selected_detection is not None
    
    def get_selected_mask(self) -> Optional[np.ndarray]:
        """Get the mask of the currently selected object.
        
        Returns:
            Segmentation mask if available, None otherwise
        """
        if self.selected_detection and self.selected_detection.mask is not None:
            return self.selected_detection.mask
        return None
    
    def get_selection_info(self) -> Optional[str]:
        """Get human-readable info about current selection.
        
        Returns:
            String with class name and confidence, or None
        """
        if self.selected_detection:
            det = self.selected_detection
            return f"{det.class_name} ({det.confidence:.0%})"
        return None
    
    @staticmethod
    def _centroid_distance(
        c1: Tuple[int, int],
        c2: Tuple[int, int]
    ) -> float:
        """Calculate Euclidean distance between two centroids."""
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
