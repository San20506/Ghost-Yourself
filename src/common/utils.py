"""Shared utility functions."""

import time
from collections import deque
from typing import Tuple

import cv2
import numpy as np


class FPSCounter:
    """Calculates and smooths FPS over time."""
    
    def __init__(self, window_size: int = 30):
        self.timestamps = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def update(self) -> float:
        """Update counter and return current FPS."""
        current_time = time.time()
        self.timestamps.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.timestamps) > 1:
            avg_time = sum(self.timestamps) / len(self.timestamps)
            return 1.0 / avg_time if avg_time > 0 else 0.0
        return 0.0


def calculate_fps(prev_time: float) -> Tuple[float, float]:
    """Calculate FPS from previous timestamp.
    
    Args:
        prev_time: Previous frame timestamp
        
    Returns:
        Tuple of (fps, current_time)
    """
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
    return fps, current_time


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    font_thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (40, 40, 40),
    padding: int = 5
) -> np.ndarray:
    """Draw text with a background rectangle for visibility.
    
    Args:
        frame: Image to draw on
        text: Text to display
        position: (x, y) position for text
        font_scale: Font size scale
        font_thickness: Font stroke thickness
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        padding: Padding around text
        
    Returns:
        Modified frame
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1  # Filled
    )
    
    # Draw text
    cv2.putText(
        frame, text, (x, y),
        font, font_scale, text_color, font_thickness, cv2.LINE_AA
    )
    
    return frame


def create_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 120, 255),
    alpha: float = 0.4
) -> np.ndarray:
    """Create a colored overlay for a mask.
    
    Args:
        frame: Original frame
        mask: Binary mask (single channel)
        color: Overlay color (BGR)
        alpha: Transparency (0-1)
        
    Returns:
        Frame with mask overlay
    """
    overlay = frame.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(frame)
    colored_mask[:] = color
    
    # Apply mask
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(
        frame[mask_bool], 1 - alpha,
        colored_mask[mask_bool], alpha,
        0
    )
    
    return overlay


def dilate_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Dilate a binary mask to expand its coverage.
    
    Args:
        mask: Binary mask
        kernel_size: Size of dilation kernel
        
    Returns:
        Dilated mask
    """
    if kernel_size <= 0:
        return mask
    
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated


def blur_mask_edges(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """Apply blur to mask edges for smoother blending.
    
    Args:
        mask: Binary mask
        blur_size: Blur kernel size (must be odd)
        
    Returns:
        Blurred mask
    """
    if blur_size <= 0:
        return mask
    
    # Ensure odd kernel size
    if blur_size % 2 == 0:
        blur_size += 1
    
    return cv2.GaussianBlur(mask.astype(np.uint8), (blur_size, blur_size), 0)


def point_in_box(
    point: Tuple[int, int],
    box: Tuple[int, int, int, int]
) -> bool:
    """Check if a point is inside a bounding box.
    
    Args:
        point: (x, y) coordinates
        box: (x1, y1, x2, y2) bounding box
        
    Returns:
        True if point is inside box
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2
