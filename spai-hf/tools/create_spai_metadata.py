#!/usr/bin/env python3
"""
Create metadata CSV for SPAI inference on crawled images.
"""

import pandas as pd
from pathlib import Path
import cv2

def create_spai_metadata(image_dir, output_csv):
    """
    Create SPAI-compatible metadata CSV.
    """
    image_dir = Path(image_dir)
    images = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    rows = []
    for idx, img_path in enumerate(images):
        try:
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
            else:
                h, w = 0, 0
            
            row = {
                'image': img_path.name,
                'split': 'test',
                'class': 0,  # Unknown (for inference)
                'width': w,
                'height': h,
                'file_path': str(img_path.relative_to(image_dir.parent))
            }
            rows.append(row)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Created {output_csv}")
    print(f"   Total images: {len(df)}")
    print(f"   Columns: {', '.join(df.columns)}")
    print(f"\n📊 Sample:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
        output_csv = sys.argv[2]
    else:
        image_dir = "output/images"
        output_csv = "output/images_spai_metadata.csv"
    
    create_spai_metadata(image_dir, output_csv)
