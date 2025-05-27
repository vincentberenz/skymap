from typing import Optional

import cv2
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from .logger import logger


def _bits_reduction(data: np.ndarray, target: np.dtype) -> np.ndarray:
    original_max = np.iinfo(data.dtype).max
    target_max = np.iinfo(target).max
    ratio = target_max / original_max
    return (data * ratio).astype(target)


def to_8bits(image: np.ndarray) -> np.ndarray:
    """
    Convert image to 8 bits (i.e. returns an array
    of dtype numpy uint8)
    """
    return _bits_reduction(image, np.dtype(np.uint8))


def load_image(image_path: str) -> np.ndarray:
    """
    Load and validate a 16-bit image

    Parameters:
    -----------
    image_path : str
        Path to the input image file

    Returns:
    --------
    np.ndarray
        Loaded image array

    Raises:
    ------
    ValueError
        If image loading fails or image is not 16-bit
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        if image.dtype != np.uint16:
            raise ValueError("Image must be 16-bit format")
        logger.info(f"Successfully loaded image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise


def visualize_healpix_map(
    healpix_map: np.ndarray, output_path: Optional[str] = None
) -> None:
    """
    Visualize the HEALPix map

    Parameters:
    -----------
    healpix_map : np.ndarray
        2D array of HEALPix indices
    output_path : str, optional
        Path to save the visualization
    """
    try:
        # Get unique HEALPix indices
        unique_indices = np.unique(healpix_map)

        # Create color map
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(unique_indices)))

        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(healpix_map, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="HEALPix Index")
        plt.title("HEALPix Map")

        if output_path:
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        raise


def plot_healpix_projection(
    healpix_map: np.ndarray, nside: Optional[int] = None
) -> None:
    """
    Plot the HEALPix map in Mollweide projection

    Parameters:
    -----------
    healpix_map : np.ndarray
        2D array of HEALPix indices
    nside : int, optional
        HEALPix resolution parameter
    """
    try:
        if nside is None:
            nside = hp.npix2nside(np.max(healpix_map) + 1)

        # Create Mollweide projection
        hp.mollview(healpix_map, nest=True, title="HEALPix Map")
        plt.show()

    except Exception as e:
        logger.error(f"Error in HEALPix projection: {str(e)}")
        raise


def validate_healpix_map(healpix_map, nside):
    """
    Validate the HEALPix map

    Parameters:
    -----------
    healpix_map : np.ndarray
        2D array of HEALPix indices
    nside : int
        HEALPix resolution parameter
    """
    try:
        if not isinstance(healpix_map, np.ndarray):
            raise ValueError("HEALPix map must be a numpy array")

        if np.any(healpix_map < 0):
            raise ValueError("HEALPix indices must be non-negative")

        max_index = hp.nside2npix(nside) - 1
        if np.any(healpix_map > max_index):
            raise ValueError(f"HEALPix indices exceed maximum for nside={nside}")

        logger.info("HEALPix map validation passed")

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise
