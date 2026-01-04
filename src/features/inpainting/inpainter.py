"""LaMa-based image inpainting for real-time object removal."""

from typing import Optional, Tuple

import cv2
import numpy as np
from simple_lama_inpainting import SimpleLama

from ...config import InpaintingConfig
from ...common.utils import dilate_mask, blur_mask_edges


class Inpainter:
    """LaMa inpainting wrapper for real-time object removal.
    
    Uses the simple-lama-inpainting package which provides a clean
    interface to the LaMa (Large Mask Inpainting) model.
    """
    
    def __init__(self, config: InpaintingConfig):
        """Initialize the inpainter.
        
        Args:
            config: Inpainting configuration
        """
        self.config = config
        
        # Initialize LaMa model (downloads weights on first run)
        print("Loading LaMa inpainting model...")
        self.model = SimpleLama()
        print("LaMa model loaded successfully!")
        
        # Background reference frame for blending
        self.background_frame: Optional[np.ndarray] = None
    
    def inpaint(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Remove an object from the frame using inpainting.
        
        Args:
            frame: BGR image from OpenCV
            mask: Binary mask where white (255) = area to remove
            
        Returns:
            Inpainted frame with object removed
        """
        if mask is None or np.sum(mask) == 0:
            return frame
        
        # Process mask
        processed_mask = self._process_mask(mask)
        
        # Optimization: Fast background replacement
        # If we have a background and the option is enabled, skip the model!
        if (self.config.fast_background_replace and 
            self.background_frame is not None):
            return self._replace_with_background(frame, processed_mask)
        
        # Convert frame from BGR to RGB for LaMa
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inpainting
        result_rgb = self.model(frame_rgb, processed_mask)
        
        # Convert back to BGR for OpenCV
        result_bgr = cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
        
        # Optional: blend with background for smoother results
        if self.config.use_background_blend and self.background_frame is not None:
            result_bgr = self._blend_with_background(
                result_bgr, processed_mask
            )
        
        return result_bgr
    
    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Process mask with dilation and blur for better inpainting.
        
        Args:
            mask: Input binary mask
            
        Returns:
            Processed mask
        """
        # Ensure binary
        processed = (mask > 127).astype(np.uint8) * 255
        
        # Dilate to cover edges around the object
        if self.config.mask_dilation > 0:
            processed = dilate_mask(processed, self.config.mask_dilation)
        
        # Optional blur for smoother transitions
        if self.config.mask_blur > 0:
            processed = blur_mask_edges(processed, self.config.mask_blur)
        
        return processed
    
    def _blend_with_background(
        self,
        inpainted: np.ndarray,
        mask: np.ndarray,
        blend_margin: int = 20
    ) -> np.ndarray:
        """Blend inpainted result with captured background.
        
        Uses the captured background in areas where we have a stable
        reference, giving better results than pure inpainting.
        
        Args:
            inpainted: Inpainted frame
            mask: The mask used for inpainting
            blend_margin: Margin for gradient blending
            
        Returns:
            Blended result
        """
        if self.background_frame is None:
            return inpainted
        
        # Resize background if needed
        bg = self.background_frame
        if bg.shape[:2] != inpainted.shape[:2]:
            bg = cv2.resize(bg, (inpainted.shape[1], inpainted.shape[0]))
        
        # Create blend mask with gradient at edges
        blend_mask = mask.copy().astype(np.float32) / 255.0
        
        # Use the inpainted result where mask is, background elsewhere
        # This gives us the best of both worlds
        result = inpainted.copy()
        mask_bool = blend_mask > 0.5
        
        # Blend: 70% inpainted, 30% background in masked region
        result[mask_bool] = cv2.addWeighted(
            inpainted[mask_bool], 0.7,
            bg[mask_bool], 0.3,
            0
        )
        
        return result
    
    def _replace_with_background(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Replace masked area directly with background (O(1) inpainting).
        
        Args:
            frame: Original frame
            mask: Processed mask
            
        Returns:
            Frame with background pasted in masked area
        """
        if self.background_frame is None:
            return frame
        
        # Resize background if needed
        bg = self.background_frame
        if bg.shape[:2] != frame.shape[:2]:
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
        
        # Normalize mask to 0-1 for blending
        mask_norm = mask.astype(np.float32) / 255.0
        mask_3d = mask_norm[:, :, np.newaxis]
        
        # Perform linear interpolation: frame * (1 - mask) + bg * mask
        # This gives seamless edges due to the blur in processed_mask
        result = (frame * (1 - mask_3d) + bg * mask_3d).astype(np.uint8)
        
        return result

    def capture_background(self, frame: np.ndarray):
        """Capture current frame as background reference.
        
        Should be called when the scene is empty (object not present).
        
        Args:
            frame: Clean background frame to store
        """
        self.background_frame = frame.copy()
        print("Background captured!")
    
    def has_background(self) -> bool:
        """Check if a background reference has been captured."""
        return self.background_frame is not None
    
    def clear_background(self):
        """Clear the captured background reference."""
        self.background_frame = None

