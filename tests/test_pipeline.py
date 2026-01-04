"""Tests for the Ghost Yourself pipeline components."""

import numpy as np
import pytest


class TestConfiguration:
    """Test configuration settings."""
    
    def test_default_config(self):
        """Verify default configuration can be created."""
        from src.config import AppConfig
        
        config = AppConfig.default()
        
        assert config.video.width == 640
        assert config.video.height == 480
        assert config.detection.confidence_threshold == 0.5
        assert config.ui.window_name == "Ghost Yourself"
    
    def test_performance_config(self):
        """Verify performance configuration has optimized settings."""
        from src.config import AppConfig
        
        config = AppConfig.performance()
        
        assert config.video.process_scale < 1.0
        assert config.video.frame_skip >= 1


class TestUtils:
    """Test utility functions."""
    
    def test_point_in_box(self):
        """Test point-in-box detection."""
        from src.common.utils import point_in_box
        
        box = (10, 10, 100, 100)
        
        # Inside
        assert point_in_box((50, 50), box) is True
        
        # On edge
        assert point_in_box((10, 10), box) is True
        
        # Outside
        assert point_in_box((5, 5), box) is False
        assert point_in_box((150, 50), box) is False
    
    def test_dilate_mask(self):
        """Test mask dilation."""
        from src.common.utils import dilate_mask
        
        # Create a small mask with a single point
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255
        
        dilated = dilate_mask(mask, kernel_size=10)
        
        # Dilated mask should have more non-zero pixels
        assert np.sum(dilated > 0) > np.sum(mask > 0)
    
    def test_fps_counter(self):
        """Test FPS counter."""
        from src.common.utils import FPSCounter
        import time
        
        counter = FPSCounter(window_size=10)
        
        # First call should return 0 or a low value
        fps1 = counter.update()
        
        # Wait a bit and call again
        time.sleep(0.1)
        fps2 = counter.update()
        
        # FPS should be around 10 (100ms delay)
        assert 5 < fps2 < 20


class TestTracker:
    """Test object tracking."""
    
    def test_select_at_point(self):
        """Test object selection by click point."""
        from src.features.detection import ObjectTracker, Detection
        
        tracker = ObjectTracker()
        
        # Create mock detections
        det1 = Detection(
            id=1,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(10, 10, 100, 100)
        )
        det2 = Detection(
            id=2,
            class_id=1,
            class_name="bicycle",
            confidence=0.8,
            bbox=(200, 200, 300, 300)
        )
        
        detections = [det1, det2]
        
        # Select first object
        selected = tracker.select_at_point((50, 50), detections)
        assert selected is not None
        assert selected.id == 1
        assert selected.class_name == "person"
        
        # Select second object
        selected = tracker.select_at_point((250, 250), detections)
        assert selected is not None
        assert selected.id == 2
        
        # Click outside - should clear
        selected = tracker.select_at_point((500, 500), detections)
        assert selected is None
        assert not tracker.has_selection()
    
    def test_track_across_frames(self):
        """Test tracking maintains selection across frames."""
        from src.features.detection import ObjectTracker, Detection
        
        tracker = ObjectTracker()
        
        # Frame 1: Select person at (50, 50)
        det1 = Detection(
            id=1,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(10, 10, 100, 100)
        )
        tracker.select_at_point((50, 50), [det1])
        assert tracker.has_selection()
        
        # Frame 2: Person has moved slightly
        det2 = Detection(
            id=99,  # New ID from detector
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(20, 20, 110, 110)  # Moved 10px
        )
        
        matched = tracker.track([det2])
        assert matched is not None
        assert matched.class_name == "person"
    
    def test_clear_selection(self):
        """Test clearing selection."""
        from src.features.detection import ObjectTracker, Detection
        
        tracker = ObjectTracker()
        
        det = Detection(
            id=1,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(10, 10, 100, 100)
        )
        
        tracker.select_at_point((50, 50), [det])
        assert tracker.has_selection()
        
        tracker.clear_selection()
        assert not tracker.has_selection()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
