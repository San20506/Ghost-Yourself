    def _replace_with_background(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Replace masked area deeply with background (O(1) inpainting).
        
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
        
        # Create result buffer
        result = frame.copy()
        
        # Normalize mask to 0-1
        mask_norm = mask.astype(np.float32) / 255.0
        
        # Expand mask slightly more for seamless blending if needed
        # But since we have a perfect background, we can just mix based on mask
        # Using soft blending at edges to avoid harsh lines
        
        mask_bool = mask_norm > 0
        
        # Where mask is 0, use frame. Where mask is >0, blend.
        # Simple implementation: direct paste + feathering
        # But we already have blurred edges in processed_mask
        
        # Vectorized blending:
        # result = frame * (1 - mask) + bg * mask
        
        # Expand dimensions for broadcasting (H, W) -> (H, W, 1)
        mask_3d = mask_norm[:, :, np.newaxis]
        
        # Perform linear interpolation
        result = (frame * (1 - mask_3d) + bg * mask_3d).astype(np.uint8)
        
        return result
