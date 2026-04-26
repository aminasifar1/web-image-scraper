#!/usr/bin/env python3
"""
Extract Fourier noise analysis from images.
Calculates spectral characteristics useful for AI-generated image detection.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_fourier_features(image_path):
    """
    Extract Fourier-based features from an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with Fourier features
    """
    try:
        # Read image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        h, w = img.shape
        
        # Apply Hanning window to reduce edge effects
        window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
        img_windowed = img * window
        
        # Compute 2D FFT
        fft = np.fft.fft2(img_windowed)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Compute phase
        phase = np.angle(fft_shift)
        
        # Average magnitude spectrum
        avg_magnitude = np.mean(magnitude)
        
        # Compute log spectrum for better visualization
        log_magnitude = np.log1p(magnitude)
        
        # Spectral distribution by radial frequency
        center_y, center_x = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        radial_distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Define frequency bands
        max_freq = np.sqrt(center_x**2 + center_y**2)
        bands = np.linspace(0, max_freq, 6)  # 5 bands
        
        energy_by_band = []
        for i in range(len(bands) - 1):
            mask = (radial_distance >= bands[i]) & (radial_distance < bands[i+1])
            energy = np.sum(magnitude[mask])
            energy_by_band.append(float(energy))
        
        # Total energy
        total_energy = np.sum(magnitude)
        
        # Normalize energy by band
        energy_ratio = [e / (total_energy + 1e-10) for e in energy_by_band]
        
        # Spectral entropy (measure of order/chaos)
        spectrum_flat = magnitude.flatten()
        spectrum_norm = spectrum_flat / (np.sum(spectrum_flat) + 1e-10)
        spectrum_entropy = -np.sum(spectrum_norm * np.log(spectrum_norm + 1e-10))
        
        # High frequency content (above 50% of max frequency)
        high_freq_mask = radial_distance > (max_freq * 0.5)
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
        
        # Low frequency content (below 20% of max frequency)
        low_freq_mask = radial_distance < (max_freq * 0.2)
        low_freq_energy = np.sum(magnitude[low_freq_mask])
        low_freq_ratio = low_freq_energy / (total_energy + 1e-10)
        
        # Spectral flatness (Wiener entropy)
        geometric_mean = np.exp(np.mean(np.log(spectrum_norm + 1e-10)))
        arithmetic_mean = np.mean(spectrum_norm)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Phase coherence (measure of structure)
        phase_coherence = np.mean(np.abs(np.sin(phase)))
        
        # Anisotropy (directional preference in spectrum)
        quadrant_energies = [
            np.sum(magnitude[0:center_y, 0:center_x]),  # TL
            np.sum(magnitude[0:center_y, center_x:]),   # TR
            np.sum(magnitude[center_y:, 0:center_x]),   # BL
            np.sum(magnitude[center_y:, center_x:])     # BR
        ]
        quadrant_ratio = np.std(quadrant_energies) / (np.mean(quadrant_energies) + 1e-10)
        
        features = {
            'avg_magnitude': float(avg_magnitude),
            'total_energy': float(total_energy),
            'spectrum_entropy': float(spectrum_entropy),
            'high_freq_ratio': float(high_freq_ratio),
            'low_freq_ratio': float(low_freq_ratio),
            'spectral_flatness': float(spectral_flatness),
            'phase_coherence': float(phase_coherence),
            'quadrant_anisotropy': float(quadrant_ratio),
            'band_0_energy': float(energy_ratio[0]) if len(energy_ratio) > 0 else 0.0,
            'band_1_energy': float(energy_ratio[1]) if len(energy_ratio) > 1 else 0.0,
            'band_2_energy': float(energy_ratio[2]) if len(energy_ratio) > 2 else 0.0,
            'band_3_energy': float(energy_ratio[3]) if len(energy_ratio) > 3 else 0.0,
            'band_4_energy': float(energy_ratio[4]) if len(energy_ratio) > 4 else 0.0,
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def process_images_with_fourier(image_dir, metadata_csv, output_csv):
    """
    Process all images in directory and extract Fourier features.
    
    Args:
        image_dir: Directory containing images
        metadata_csv: CSV with image metadata
        output_csv: Output CSV path
    """
    image_dir = Path(image_dir)
    
    # Read metadata
    df = pd.read_csv(metadata_csv)
    logger.info(f"📖 Loaded {len(df)} images from metadata")
    
    # Extract Fourier features
    fourier_features = []
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Fourier features"):
        # Try both possible path column names
        image_path = None
        if 'filtered_path' in row:
            original_path = row['filtered_path']
            # Replace old path with new one
            if 'elpais_crawler' in str(original_path):
                image_path = Path(str(original_path).replace('output/elpais_crawler', 'output/crawler_output'))
            else:
                image_path = Path(original_path)
        elif 'file_path' in row:
            image_path = Path(row['file_path'])
        else:
            failed += 1
            continue
        
        # If still not found, try looking in raw_images
        if not image_path.exists():
            filename = Path(image_path).name
            alt_path = image_dir / filename
            if alt_path.exists():
                image_path = alt_path
            else:
                failed += 1
                continue
        
        if not image_path.exists():
            failed += 1
            continue
        
        features = extract_fourier_features(image_path)
        
        if features is None:
            failed += 1
            continue
        
        # Combine metadata with Fourier features
        result = {
            'filename': row.get('filename', image_path.name),
            'filtered_path': str(image_path),
            'image_url': row.get('image_url', ''),
            'declared_width': row.get('declared_width', 0),
            'declared_height': row.get('declared_height', 0),
            'category': row.get('category', ''),
            'subcategory': row.get('subcategory', ''),
            'seed_url': row.get('seed_url', ''),
            **features
        }
        
        fourier_features.append(result)
    
    # Save to CSV
    result_df = pd.DataFrame(fourier_features)
    result_df.to_csv(output_csv, index=False)
    
    logger.info(f"✅ Processed {len(fourier_features)} images successfully")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📁 Results saved to {output_csv}")
    
    return result_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
        metadata_csv = sys.argv[2]
        output_csv = sys.argv[3]
    else:
        image_dir = "output/crawler_output/images"
        metadata_csv = "output/crawler_output/metadata/images.csv"
        output_csv = "output/fourier_analysis.csv"
    
    process_images_with_fourier(image_dir, metadata_csv, output_csv)
