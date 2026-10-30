"""
OpenCV Image Preprocessing Module

Provides CLAHE, resizing, and other preprocessing operations
for vehicle images before classification and OCR.
"""

import cv2
import numpy as np


class OpenCVPreprocessor:
    """OpenCV-based image preprocessing for ANPR and classification."""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), 
                 target_size=(640, 640)):
        """
        Initialize the preprocessor.
        
        Args:
            clip_limit: CLAHE clip limit for contrast enhancement
            tile_grid_size: CLAHE tile grid size
            target_size: Target size for resizing (width, height)
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.target_size = target_size
    
    def apply_clahe(self, bgr_image):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            bgr_image: Input image in BGR format (OpenCV default)
            
        Returns:
            CLAHE-enhanced BGR image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                tileGridSize=self.tile_grid_size)
        l_enhanced = clahe.apply(l_channel)
        
        # Merge channels back
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def resize_image(self, image, target_size=None):
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image
            target_size: Target size (width, height), uses self.target_size if None
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    def preprocess(self, bgr_image, apply_clahe=True, resize=True):
        """
        Full preprocessing pipeline.
        
        Args:
            bgr_image: Input BGR image
            apply_clahe: Whether to apply CLAHE
            resize: Whether to resize to target dimensions
            
        Returns:
            Preprocessed image ready for further processing
        """
        result = bgr_image.copy()
        
        if apply_clahe:
            result = self.apply_clahe(result)
        
        if resize:
            result = self.resize_image(result)
        
        return result
    
    def convert_to_gray(self, image):
        """
        Convert image to grayscale.
        
        Args:
            image: BGR or RGB image
            
        Returns:
            Grayscale image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def normalize_image(self, image):
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image with float values in [0, 1]
        """
        return image.astype(np.float32) / 255.0


# Convenience function for quick preprocessing
def quick_preprocess(image, clip_limit=2.0, target_size=(640, 640)):
    """
    Quick preprocessing pipeline for a single image.
    
    Args:
        image: BGR image array
        clip_limit: CLAHE clip limit
        target_size: Target resize dimensions
        
    Returns:
        Preprocessed image
    """
    preprocessor = OpenCVPreprocessor(clip_limit=clip_limit, target_size=target_size)
    return preprocessor.preprocess(image)