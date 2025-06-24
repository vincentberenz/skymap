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

# to update: mix of patch sizes depending on the image area
PATCH_SIZE = [300,300]

WORKING_DIR = Path("/tmp/skymap_debug/")
NO_PLATE_SOLVING = False
CPULIMIT_SECONDS = 5
NUM_PROCESSES = 8


def main():
    """Process the image and save the result."""
    try:
        logger.info(f"Processing image: {INPUT_IMAGE}")
        logger.info(f"Patch size: {PATCH_SIZE}")

        # Ensure output directory exists
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Load and process the image
        patched_image = PatchedImage.from_file(
            INPUT_IMAGE,
            patch_size=PATCH_SIZE,
            num_processes=NUM_PROCESSES,
            working_dir=WORKING_DIR,
            no_plate_solving=NO_PLATE_SOLVING,
            cpulimit_seconds=CPULIMIT_SECONDS,
        )

        # Save the processed image
        logger.info(f"Saving processed image to: {OUTPUT_PATH}")
        patched_image.dump(OUTPUT_PATH)
        logger.success(f"Successfully saved processed image to {OUTPUT_PATH}")

        # Create and save visualization
        vis_path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_visualization.tiff")
        logger.info(f"Creating visualization: {vis_path}")
        patched_image.display(vis_path)
        logger.success(f"Successfully saved visualization to {vis_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        import traceback
        logger.error("Traceback (most recent call last):\n" + traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
