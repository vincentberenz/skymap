#!/usr/bin/env python3
"""
Script to process a TIFF image into a PatchedImage and save it.

This script loads a TIFF image, processes it into patches with plate solving,
and saves the resulting PatchedImage to a file.
"""

import sys
from pathlib import Path

from loguru import logger

from skymapper.patches import PatchedImage

# Configuration
INPUT_IMAGE = Path("images/nightskycam3_2025_04_05_03_54_30.tiff")
OUTPUT_PATH = INPUT_IMAGE.with_suffix(".pkl.gz")
PATCH_SIZE = 400
PATCH_OVERLAP = 20


def main():
    """Process the image and save the result."""
    try:
        logger.info(f"Processing image: {INPUT_IMAGE}")
        logger.info(f"Patch size: {PATCH_SIZE}, Overlap: {PATCH_OVERLAP}")

        # Ensure output directory exists
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Load and process the image
        patched_image = PatchedImage.from_file(
            INPUT_IMAGE,
            patch_size=PATCH_SIZE,
            patch_overlap=PATCH_OVERLAP,
            num_processes=1,
        )

        # Save the processed image
        logger.info(f"Saving processed image to: {OUTPUT_PATH}")
        patched_image.dump(OUTPUT_PATH)
        logger.success(f"Successfully saved processed image to {OUTPUT_PATH}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.trace("Full traceback:")
        import traceback

        logger.trace(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
