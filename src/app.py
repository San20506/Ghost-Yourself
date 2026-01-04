"""
Ghost Yourself - Real-Time Object Removal Video Stream

A computer vision application that makes objects—or yourself—disappear
from a live camera feed in real time using YOLOv8 segmentation and LaMa inpainting.

Controls:
    Click - Select/deselect object to remove
    d     - Toggle detection visualization
    b     - Capture background reference
    r     - Reset selection
    q/ESC - Quit application
"""

import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import AppConfig
from .common.utils import FPSCounter, draw_text_with_background, create_mask_overlay
from .features.detection import ObjectDetector, ObjectTracker
from .features.inpainting import Inpainter
from .features.video import VideoPipeline


class GhostYourselfApp:
    """Main application class for real-time object removal."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the application.
        
        Args:
            config: Application configuration. If None, uses defaults.
        """
        self.config = config or AppConfig.default()
        
        # Initialize components
        print("\n" + "="*50)
        print("  👻 Ghost Yourself - Initializing...")
        print("="*50 + "\n")
        
        self.video = VideoPipeline(self.config.video)
        self.detector = ObjectDetector(self.config.detection)
        self.tracker = ObjectTracker()
        self.inpainter = Inpainter(self.config.inpainting)
        
        # State
        self.fps_counter = FPSCounter()
        self.show_detections = self.config.show_detections
        self.running = False
        
        # Mouse callback state
        self.click_point: Optional[Tuple[int, int]] = None
        
        print("\n" + "="*50)
        print("  ✅ Initialization complete!")
        print("="*50 + "\n")
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        """Handle mouse events for object selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
    
    def _draw_ui(
        self,
        frame: np.ndarray,
        fps: float,
        selection_info: Optional[str] = None
    ) -> np.ndarray:
        """Draw UI overlays on frame.
        
        Args:
            frame: Frame to draw on
            fps: Current FPS
            selection_info: Selected object info string
            
        Returns:
            Frame with UI overlays
        """
        ui = self.config.ui
        h, w = frame.shape[:2]
        
        # FPS counter (top-left)
        if ui.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            draw_text_with_background(
                frame, fps_text, (10, 25),
                font_scale=ui.font_scale,
                font_thickness=ui.font_thickness,
                text_color=ui.color_text,
                bg_color=ui.color_background
            )
        
        # Selection info (top-center)
        if selection_info:
            sel_text = f"🎯 Removing: {selection_info}"
            text_width = cv2.getTextSize(
                sel_text, cv2.FONT_HERSHEY_SIMPLEX, ui.font_scale, ui.font_thickness
            )[0][0]
            draw_text_with_background(
                frame, sel_text, ((w - text_width) // 2, 25),
                font_scale=ui.font_scale,
                font_thickness=ui.font_thickness,
                text_color=(100, 255, 100),  # Green
                bg_color=ui.color_background
            )
        
        # Controls reminder (bottom)
        if ui.show_controls:
            controls = "Click: Select | D: Detection | B: Background | R: Reset | Q: Quit"
            draw_text_with_background(
                frame, controls, (10, h - 15),
                font_scale=0.45,
                font_thickness=1,
                text_color=(180, 180, 180),
                bg_color=ui.color_background
            )
        
        # Background indicator (top-right)
        if self.inpainter.has_background():
            bg_text = "BG: ✓"
            draw_text_with_background(
                frame, bg_text, (w - 70, 25),
                font_scale=ui.font_scale,
                font_thickness=ui.font_thickness,
                text_color=(100, 255, 100),
                bg_color=ui.color_background
            )
        
        return frame
    
    def run(self):
        """Run the main application loop."""
        # Start video capture
        if not self.video.start():
            print("Failed to start video capture!")
            return
        
        # Create window
        cv2.namedWindow(self.config.ui.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.config.ui.window_name, self._mouse_callback)
        
        print("\n" + "="*50)
        print("  📹 Camera started! Controls:")
        print("    Click - Select object to remove")
        print("    D     - Toggle detection boxes")
        print("    B     - Capture background")
        print("    R     - Reset selection")
        print("    Q/ESC - Quit")
        print("="*50 + "\n")
        
        self.running = True
        last_detections = []
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.video.read()
                if not ret or frame is None:
                    print("Failed to read frame")
                    break
                
                # Update FPS
                fps = self.fps_counter.update()
                
                # Run detection (with optional frame skip)
                if self.video.should_process():
                    last_detections = self.detector.detect(frame)
                
                # Handle click for selection
                if self.click_point is not None:
                    self.tracker.select_at_point(self.click_point, last_detections)
                    self.click_point = None
                
                # Track selected object across frames
                if self.tracker.has_selection():
                    self.tracker.track(last_detections)
                
                # Get current selected mask
                mask = self.tracker.get_selected_mask()
                
                # Perform inpainting if we have a selection
                display_frame = frame.copy()
                
                if mask is not None:
                    # Inpaint to remove selected object
                    display_frame = self.inpainter.inpaint(frame, mask)
                
                # Draw detections if enabled
                if self.show_detections and last_detections:
                    selected_id = (self.tracker.selected_detection.id 
                                   if self.tracker.selected_detection else None)
                    display_frame = self.detector.draw_detections(
                        display_frame,
                        last_detections,
                        selected_id=selected_id,
                        color=self.config.ui.color_detection,
                        selected_color=self.config.ui.color_selected,
                        thickness=self.config.ui.detection_box_thickness
                    )
                
                # Draw UI overlays
                display_frame = self._draw_ui(
                    display_frame,
                    fps,
                    self.tracker.get_selection_info()
                )
                
                # Display
                cv2.imshow(self.config.ui.window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key, frame)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self._cleanup()
    
    def _handle_key(self, key: int, frame: np.ndarray):
        """Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey
            frame: Current frame (for background capture)
        """
        if key == ord('q') or key == 27:  # q or ESC
            self.running = False
        
        elif key == ord('d'):
            self.show_detections = not self.show_detections
            status = "ON" if self.show_detections else "OFF"
            print(f"Detection display: {status}")
        
        elif key == ord('b'):
            self.inpainter.capture_background(frame)
        
        elif key == ord('r'):
            self.tracker.clear_selection()
            print("Selection cleared")
    
    def _cleanup(self):
        """Clean up resources."""
        print("\nShutting down...")
        self.video.stop()
        cv2.destroyAllWindows()
        print("Goodbye! 👻")


def main():
    """Entry point for the application."""
    # Parse command line arguments for configuration
    config = AppConfig.default()
    
    # Check for performance mode flag
    if "--performance" in sys.argv or "-p" in sys.argv:
        print("Running in performance mode")
        config = AppConfig.performance()
    
    # Check for debug flag
    if "--debug" in sys.argv:
        config.debug = True
    
    # Create and run application
    app = GhostYourselfApp(config)
    app.run()


if __name__ == "__main__":
    main()
