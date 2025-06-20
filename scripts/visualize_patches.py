#!/usr/bin/env python3
"""
Script to visualize a previously saved PatchedImage.

This script loads a saved PatchedImage from a file and generates
a visualization showing the patch borders (green for solved, red for unsolved).
"""

from pathlib import Path
from skymapper.patches import PatchedImage
from loguru import logger

# Configuration - Update this path to point to your saved PatchedImage file
INPUT_FILE = Path("images/nightskycam3_2025_04_05_03_54_30.pkl.gz")


def main():
    """Load a PatchedImage and generate its visualization."""
    try:
        logger.info(f"Loading PatchedImage from: {INPUT_FILE}")
        
        patched_image = PatchedImage.load(INPUT_FILE)
       
        logger.info(f"Loaded PatchedImage with {len(patched_image.patches)} patches and image of shape {patched_image.image.shape}")

        # Generate visualization path
        vis_path = INPUT_FILE.with_name(f"{INPUT_FILE.stem}_visualization.tiff")
        logger.info(f"Creating visualization: {vis_path}")
        
        # Generate and save the visualization
        patched_image.display(vis_path)
        logger.success(f"Successfully saved visualization to {vis_path}")
        
        return 0
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {INPUT_FILE}")
        return 1
    except Exception as e:
        logger.error(f"Error processing {INPUT_FILE}: {e}")
        import traceback
        logger.error("Traceback (most recent call last):\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
