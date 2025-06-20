import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet
from loguru import logger

# Astrometry.net API key
ASTROMETRY_NET_API_KEY = "xyryxekynhqcobhx"

# Constants
ROOT_DIR = Path(__file__).parent
IMAGE_PATH = ROOT_DIR / "nightskycam5_2025_04_30_23_42_30.tiff"
SUBMITTED_IMAGE = ROOT_DIR / "astrometry_submitted.tiff"
LOG_LEVEL = "DEBUG"
LOG_FORMAT = "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
LOG_FILE = Path(f"astrometry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Processing settings
PATCH_SIZE = 2000  # Size of the central patch to extract in pixels
ASTROMETRY_TIMEOUT = 300  # 5 minutes timeout for plate solving

# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level=LOG_LEVEL, format=LOG_FORMAT, colorize=True)
logger.info("Astrometry.net plate solver initialized")


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize image data to 8-bit range (0-255)"""
    if data.dtype == np.uint8:
        return data

    # Handle uint16 or other types
    if data.dtype == np.uint16:
        # Scale 16-bit to 8-bit
        return (data // 256).astype(np.uint8)

    # For other types, normalize to 0-255 range
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:
        data = (data - data_min) * 255.0 / (data_max - data_min)
    return data.astype(np.uint8)


def load_and_process_fisheye_image(
    image_path: Union[str, Path], patch_size: int
) -> np.ndarray:
    """
    Load the TIFF image, convert to grayscale if needed, and extract central square region

    Args:
        image_path: Path to the TIFF image file
        patch_size: Size of the central square patch to extract (in pixels)

    Returns:
        Processed 2D grayscale image data as numpy array (uint8)
    """
    logger.info(f"Loading image from {image_path}")
    try:
        # Load the image using imageio
        data: np.ndarray = imageio.imread(image_path)
        logger.debug(f"Loaded image with shape: {data.shape}, dtype: {data.dtype}")

        # Convert to grayscale if image is color (3D array)
        if len(data.shape) == 3:
            logger.debug("Converting color image to grayscale")
            data = data.mean(axis=2)

        # Ensure we have a 2D array
        if len(data.shape) != 2:
            error_msg = f"Expected 2D array, got {len(data.shape)}D array"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Normalize to uint8
        logger.debug("Normalizing image data to uint8")
        data = normalize_to_uint8(data)

        # Extract central square patch (up to 1000x1000 pixels)
        size: int = min(
            patch_size, min(data.shape) // 2
        )  # Ensure we don't exceed image bounds
        center: Tuple[int, int] = (data.shape[0] // 2, data.shape[1] // 2)
        logger.debug(
            f"Extracting central patch of size {size}x{size} from center {center}"
        )

        patch: Cutout2D = Cutout2D(data, center, size, mode="trim")
        logger.info(f"Successfully extracted patch with shape: {patch.data.shape}")

        data = normalize_to_uint8(patch.data)

        # Save the patch as TIFF
        output_path = SUBMITTED_IMAGE
        imageio.imsave(output_path, data, format="tiff")
        logger.info(f"Saved processed patch to {output_path}")

        return data

    except Exception as e:
        logger.exception("Error processing image")
        raise


def solve_field() -> Optional[WCS]:
    """
    Solve the astrometric solution for an image using astrometry.net

    Returns:
        WCS object if successful, None otherwise
    """
    try:
        # Initialize astrometry.net client
        ast = AstrometryNet()
        ast.api_key = ASTROMETRY_NET_API_KEY

        # Print allowed settings for debugging
        logger.info("Allowed settings for astrometry.net:")
        ast.show_allowed_settings()

        # Try to solve the image with minimal parameters first
        logger.info("Submitting to astrometry.net...")

        try:
            result = ast.solve_from_image(
                str(SUBMITTED_IMAGE),  # Ensure it's a string path
            )

            logger.info(f"API response: {result}")

            if result is not None and "wcs" in result:
                logger.success("Successfully solved the field with minimal parameters")
                return WCS(result["wcs"])

            logger.error("Failed to solve field with any parameter set")
            return None

        except Exception as e:
            logger.exception(f"Error during plate solving: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Error during plate solving: {str(e)}")
        return None


def main():
    """Main function to process the image and solve the field"""
    try:
        logger.info(f"Loading image from {IMAGE_PATH}")

        image_data = load_and_process_fisheye_image(IMAGE_PATH, PATCH_SIZE)
        logger.info(f"Original image shape: {image_data.shape}")

        # Solve the field
        logger.info("Starting plate solving...")
        wcs = solve_field()

        if wcs is not None:
            logger.success("Plate solving successful!")
            logger.info(
                f"Image center: {wcs.pixel_to_world(image_data.shape[1]/2, image_data.shape[0]/2)}"
            )
            return 0
        else:
            logger.error("Failed to solve the field")
            return 1

    except Exception as e:
        logger.exception("An error occurred during processing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
