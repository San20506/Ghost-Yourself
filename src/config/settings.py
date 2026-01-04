"""Application configuration using dataclasses."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VideoConfig:
    """Video capture and processing settings."""
    
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    
    # Performance optimization
    process_scale: float = 1.0  # Scale factor for processing (lower = faster)
    frame_skip: int = 0  # Skip N frames between processing (0 = process all)


@dataclass
class DetectionConfig:
    """YOLOv8 detection settings."""
    
    model_name: str = "yolov8n-seg.pt"  # Nano model for speed
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    # Classes to detect (None = all classes)
    # Common: 0=person, 1=bicycle, 2=car, 15=cat, 16=dog
    target_classes: list = None
    
    # Device selection
    device: str = ""  # "" = auto (CUDA if available, else CPU)


@dataclass
class InpaintingConfig:
    """LaMa inpainting settings."""
    
    # Mask processing
    mask_dilation: int = 20  # Increased for better coverage (was 15)
    mask_blur: int = 7  # Increased slightly (was 5)
    
    # Optimization
    # If True, uses captured background directly (O(1)) instead of running model
    # when a background is available. This is HUGE for efficiency.
    fast_background_replace: bool = True
    
    # Quality vs speed tradeoff
    use_background_blend: bool = True  # Blend with captured background


@dataclass
class UIConfig:
    """User interface settings."""
    
    window_name: str = "Ghost Yourself"
    
    # Colors (BGR format for OpenCV)
    color_detection: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_selected: Tuple[int, int, int] = (0, 120, 255)  # Orange
    color_text: Tuple[int, int, int] = (255, 255, 255)  # White
    color_background: Tuple[int, int, int] = (40, 40, 40)  # Dark gray
    
    # Font settings
    font_scale: float = 0.6
    font_thickness: int = 2
    
    # Overlay settings
    detection_box_thickness: int = 2
    mask_alpha: float = 0.4  # Transparency for mask overlay
    
    # Status bar
    show_fps: bool = True
    show_controls: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Debug mode
    debug: bool = False
    show_detections: bool = True  # Toggle with 'd' key
    
    @classmethod
    def default(cls) -> "AppConfig":
        """Create default configuration."""
        return cls()
    
    @classmethod
    def performance(cls) -> "AppConfig":
        """Create performance-optimized configuration."""
        return cls(
            video=VideoConfig(
                width=480,
                height=360,
                process_scale=0.75,
                frame_skip=1
            ),
            detection=DetectionConfig(
                confidence_threshold=0.6,
            ),
            inpainting=InpaintingConfig(
                mask_dilation=10,
                use_background_blend=False
            )
        )
