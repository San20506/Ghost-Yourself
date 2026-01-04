"""Detection submodule."""

from .detector import ObjectDetector, Detection
from .tracker import ObjectTracker

__all__ = ["ObjectDetector", "Detection", "ObjectTracker"]
