#!/usr/bin/env python3
"""
Convert a 16-bit RGB image to an 8-bit grayscale JPEG with a central 3500x2000 crop.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Configuration
INPUT_IMAGE = Path(__file__).parent / "nightskycam5_2025_04_30_23_42_30.tiff"
OUTPUT_IMAGE = Path(__file__).parent / "nightskycam5_2025_04_30_23_42_30.jpg"
TARGET_SIZE = (2000, 2000)  # width, height


def convert_image(
    input_path: str,
    output_path: str,
) -> None:
    """
    Convert a 16-bit RGB image to an 8-bit grayscale JPEG with a central crop.

    Args:
        input_path: Path to the input image (16-bit RGB)
        output_path: Path to save the output JPEG (optional)
        target_size: Tuple of (width, height) for the output image
    """
    # Set default output path if not provided
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"{input_stem}.jpg"

    # Read the input image
    print(f"Reading image: {input_path}")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    # Print input image properties
    print(f"Input image shape: {img.shape}, dtype: {img.dtype}")

    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        print("Converting to grayscale...")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get center crop
    height, width = img.shape
    target_width, target_height = TARGET_SIZE

    # Ensure target size is not larger than original
    target_width = min(target_width, width)
    target_height = min(target_height, height)

    # Calculate crop coordinates (centered)
    start_x = (width - target_width) // 2
    start_y = (height - target_height) // 2

    print(f"Cropping to {target_width}x{target_height} from center...")
    cropped = img[start_y : start_y + target_height, start_x : start_x + target_width]

    # Convert to 8-bit if needed
    if cropped.dtype != np.uint8:
        print("Converting to 8-bit...")
        # Scale from 16-bit to 8-bit
        cropped = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Save as JPEG
    print(f"Saving output to: {output_path}")
    cv2.imwrite(output_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("Conversion complete!")


def main():
    print(f"Processing image: {INPUT_IMAGE}")
    print(f"Output will be saved to: {OUTPUT_IMAGE}")

    try:
        convert_image(str(INPUT_IMAGE), str(OUTPUT_IMAGE))
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
