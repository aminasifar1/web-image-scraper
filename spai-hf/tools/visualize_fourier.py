#!/usr/bin/env python3
"""
Visualize Fourier noise spectrum of images.
Creates heatmaps showing frequency distribution for each image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from tqdm import tqdm
import logging

matplotlib.use('Agg')  # Non-interactive backend
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_and_visualize_fourier(image_path, output_dir, filename_only=False):
    """
    Extract Fourier spectrum and create visualization.
    
    Args:
        image_path: Path to image
        output_dir: Directory to save visualization
        filename_only: If True, use just filename without subtitle
        
    Returns:
        Path to saved visualization or None if failed
    """
    try:
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        img = img.astype(np.float32)
        
        # Apply windowing
        h, w = img.shape
        window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        img_windowed = img * window
        
        # Compute 2D FFT
        fft = np.fft.fft2(img_windowed)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Log scale for better visualization
        magnitude_log = np.log1p(magnitude)
        
        # Normalize for display
        magnitude_norm = (magnitude_log - magnitude_log.min()) / (magnitude_log.max() - magnitude_log.min() + 1e-10)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Fourier spectrum heatmap
        im = axes[1].imshow(magnitude_norm, cmap='hot')
        axes[1].set_title('Fourier Spectrum (Log Scale)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Energy (log)')
        
        # Save
        output_path = Path(output_dir) / f"{Path(image_path).stem}_fourier.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def process_all_images(image_dir, output_dir, max_images=None, sample_mode=False):
    """
    Process all images and create Fourier visualizations.
    
    Args:
        image_dir: Directory with source images
        output_dir: Directory to save visualizations
        max_images: Max images to process (None for all)
        sample_mode: If True, process every Nth image
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    images = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    if max_images:
        if sample_mode:
            # Process every Nth image
            step = max(1, len(images) // max_images)
            images = images[::step][:max_images]
        else:
            images = images[:max_images]
    
    logger.info(f"Processing {len(images)} images...")
    
    success = 0
    failed = 0
    
    for img_path in tqdm(images, desc="Creating Fourier visualizations"):
        result = extract_and_visualize_fourier(img_path, output_dir)
        if result:
            success += 1
        else:
            failed += 1
    
    logger.info(f"✅ Success: {success}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📁 Visualizations saved to {output_dir}")
    
    return success, failed


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
        output_dir = sys.argv[2]
        max_images = int(sys.argv[3]) if len(sys.argv) > 3 else None
    else:
        image_dir = "output/images"
        output_dir = "output/fourier_visualizations"
        max_images = 50  # Default to 50 for speed
    
    process_all_images(image_dir, output_dir, max_images=max_images, sample_mode=True)
