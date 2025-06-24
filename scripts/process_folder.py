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
INPUT_FOLDER = Path("images/full_night/")
OUTPUT_DIR = INPUT_FOLDER / "full_night"
EXTENSION = "tiff"

# to update: mix of patch sizes depending on the image area
PATCH_SIZE = [500,500]

WORKING_DIR = Path("/tmp/skymap_full_night/")
NO_PLATE_SOLVING = False
CPULIMIT_SECONDS = 5
NUM_PROCESSES = 7


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    PatchedImage.from_folder(
        INPUT_FOLDER,
        PATCH_SIZE,
        WORKING_DIR,
        OUTPUT_DIR,
        num_processes=NUM_PROCESSES,
        no_plate_solving=NO_PLATE_SOLVING,
        cpulimit_seconds=CPULIMIT_SECONDS,
        file_extention=EXTENSION
    )
 

if __name__ == "__main__":
    sys.exit(main())
