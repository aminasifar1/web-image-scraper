#!/usr/bin/env python3
"""
Advanced Fourier noise visualization and extraction.
Shows original, spectrum, phase, and reconstructed noise.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from tqdm import tqdm
import logging

matplotlib.use('Agg')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_noise_components(image_path):
    """
    Extract and separate noise components.
    
    Args:
        image_path: Path to image
        
    Returns:
        Dictionary with magnitude, phase, and noise
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if img is None:
            return None
        
        h, w = img.shape
        
        # Apply window
        window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        img_windowed = img * window
        
        # FFT
        fft = np.fft.fft2(img_windowed)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # High-pass filter for noise extraction
        # Create Butterworth high-pass filter
        center_y, center_x = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Cutoff frequency (high-pass)
        cutoff = min(h, w) // 4  # Keep only high frequencies
        order = 2
        H = 1 / (1 + (distance / (cutoff + 1))**(2*order))
        
        # Extract high frequencies (noise)
        magnitude_highpass = magnitude * (1 - H)
        
        # Reconstruct noise
        fft_noise = magnitude_highpass * np.exp(1j * phase)
        noise = np.fft.ifft2(fft_noise).real
        noise = np.clip(noise, 0, 255)
        
        # Low-pass component (structure)
        magnitude_lowpass = magnitude * H
        fft_lowpass = magnitude_lowpass * np.exp(1j * phase)
        structure = np.fft.ifft2(fft_lowpass).real
        structure = np.clip(structure, 0, 255)
        
        return {
            'original': img,
            'structure': structure,
            'noise': noise,
            'magnitude': magnitude,
            'phase': phase,
            'magnitude_highpass': magnitude_highpass,
            'magnitude_lowpass': magnitude_lowpass
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def visualize_noise_decomposition(image_path, output_dir):
    """
    Create comprehensive 4-panel visualization.
    """
    try:
        components = extract_noise_components(image_path)
        if not components:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        filename = Path(image_path).stem
        fig.suptitle(f'Fourier Noise Analysis: {filename}', fontsize=14, fontweight='bold')
        
        # Original
        axes[0, 0].imshow(components['original'], cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Low-freq structure
        axes[0, 1].imshow(components['structure'], cmap='gray')
        axes[0, 1].set_title('Low Frequencies (Structure)')
        axes[0, 1].axis('off')
        
        # High-freq noise
        axes[0, 2].imshow(components['noise'], cmap='hot')
        axes[0, 2].set_title('High Frequencies (Noise)')
        axes[0, 2].axis('off')
        
        # Magnitude spectrum
        mag_log = np.log1p(components['magnitude'])
        mag_norm = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-10)
        im1 = axes[1, 0].imshow(np.fft.fftshift(mag_norm), cmap='viridis')
        axes[1, 0].set_title('Magnitude Spectrum (Log)')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        # High-pass magnitude
        mag_hp_log = np.log1p(components['magnitude_highpass'])
        mag_hp_norm = (mag_hp_log - mag_hp_log.min()) / (mag_hp_log.max() - mag_hp_log.min() + 1e-10)
        im2 = axes[1, 1].imshow(np.fft.fftshift(mag_hp_norm), cmap='hot')
        axes[1, 1].set_title('High-Pass Filter')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        # Low-pass magnitude
        mag_lp_log = np.log1p(components['magnitude_lowpass'])
        mag_lp_norm = (mag_lp_log - mag_lp_log.min()) / (mag_lp_log.max() - mag_lp_log.min() + 1e-10)
        im3 = axes[1, 2].imshow(np.fft.fftshift(mag_lp_norm), cmap='viridis')
        axes[1, 2].set_title('Low-Pass Filter')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
        
        plt.tight_layout()
        output_path = Path(output_dir) / f"{filename}_noise_decomposition.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error visualizing {image_path}: {e}")
        return None


def process_all_with_decomposition(image_dir, output_dir, max_images=None):
    """
    Process all images with full decomposition.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    images = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    if max_images:
        step = max(1, len(images) // max_images)
        images = images[::step][:max_images]
    
    logger.info(f"Processing {len(images)} images with full decomposition...")
    
    success = 0
    for img_path in tqdm(images, desc="Decomposing noise"):
        result = visualize_noise_decomposition(img_path, output_dir)
        if result:
            success += 1
    
    logger.info(f"✅ Created {success} visualizations in {output_dir}")
    return success


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
        output_dir = sys.argv[2]
        max_images = int(sys.argv[3]) if len(sys.argv) > 3 else None
    else:
        image_dir = "output/images"
        output_dir = "output/noise_decomposition"
        max_images = 30
    
    process_all_with_decomposition(image_dir, output_dir, max_images=max_images)
