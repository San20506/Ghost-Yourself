"""Video capture and processing pipeline."""

from typing import Optional, Generator, Tuple

import cv2
import numpy as np

from ...config import VideoConfig


class VideoPipeline:
    """Handles webcam capture and frame processing.
    
    Provides an interface for reading frames from the webcam
    with optional preprocessing for performance optimization.
    """
    
    def __init__(self, config: VideoConfig):
        """Initialize the video pipeline.
        
        Args:
            config: Video configuration
        """
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        
        # Store original dimensions
        self.original_width = config.width
        self.original_height = config.height
        
        # Calculate processed dimensions
        self.process_width = int(config.width * config.process_scale)
        self.process_height = int(config.height * config.process_scale)
    
    def start(self) -> bool:
        """Start video capture.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.config.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.config.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera started: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        # Update dimensions if camera doesn't support requested
        self.original_width = actual_width
        self.original_height = actual_height
        self.process_width = int(actual_width * self.config.process_scale)
        self.process_height = int(actual_height * self.config.process_scale)
        
        return True
    
    def stop(self):
        """Stop video capture and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame_count = 0
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        
        return ret, frame
    
    def read_processed(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read and preprocess a frame for detection.
        
        If process_scale < 1.0, returns a smaller frame for
        faster detection while keeping the original for display.
        
        Returns:
            Tuple of (success, original_frame, processed_frame)
        """
        ret, frame = self.read()
        
        if not ret or frame is None:
            return False, None, None
        
        # If processing at full resolution, return same frame
        if self.config.process_scale >= 1.0:
            return True, frame, frame
        
        # Resize for processing
        processed = cv2.resize(
            frame,
            (self.process_width, self.process_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        return True, frame, processed
    
    def should_process(self) -> bool:
        """Check if current frame should be processed.
        
        Uses frame_skip setting to skip frames for performance.
        
        Returns:
            True if frame should be processed
        """
        if self.config.frame_skip <= 0:
            return True
        return self.frame_count % (self.config.frame_skip + 1) == 0
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously.
        
        Yields:
            Frames from camera
        """
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame
    
    def is_running(self) -> bool:
        """Check if video capture is active."""
        return self.cap is not None and self.cap.isOpened()
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get current frame dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        return self.original_width, self.original_height
    
    def get_process_dimensions(self) -> Tuple[int, int]:
        """Get processing frame dimensions.
        
        Returns:
            Tuple of (width, height) for processed frames
        """
        return self.process_width, self.process_height
    
    def scale_point(
        self,
        point: Tuple[int, int],
        from_processed: bool = True
    ) -> Tuple[int, int]:
        """Scale a point between original and processed coordinates.
        
        Args:
            point: (x, y) coordinates
            from_processed: If True, scale from processed to original.
                          If False, scale from original to processed.
                          
        Returns:
            Scaled (x, y) coordinates
        """
        x, y = point
        
        if self.config.process_scale >= 1.0:
            return point
        
        if from_processed:
            # Processed -> Original
            scale = 1.0 / self.config.process_scale
        else:
            # Original -> Processed
            scale = self.config.process_scale
        
        return int(x * scale), int(y * scale)
    
    def scale_mask(
        self,
        mask: np.ndarray,
        to_original: bool = True
    ) -> np.ndarray:
        """Scale a mask between original and processed dimensions.
        
        Args:
            mask: Input mask
            to_original: If True, scale to original dimensions.
                        If False, scale to processed dimensions.
                        
        Returns:
            Scaled mask
        """
        if self.config.process_scale >= 1.0:
            return mask
        
        if to_original:
            target_size = (self.original_width, self.original_height)
        else:
            target_size = (self.process_width, self.process_height)
        
        return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
