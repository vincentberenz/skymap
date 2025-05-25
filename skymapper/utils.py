import logging
from typing import Optional

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from .mapper import convert_to_healpix

logger = logging.getLogger(__name__)


def visualize_healpix_map(healpix_map: np.ndarray, output_path: Optional[str] = None) -> None:
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


def plot_healpix_projection(healpix_map: np.ndarray, nside: Optional[int] = None) -> None:
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
