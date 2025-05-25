import logging
from typing import Optional
import cv2
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import griddata
from astroquery.astrometry_net import AstrometryNet
from typing import Optional

import logging
from .calibration import load_calibration, correct_distortion
from .plate_solver import solve_plate

logger = logging.getLogger(__name__)

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

def correct_fisheye_distortion(image: np.ndarray, calibration_file: str) -> np.ndarray:
    """
    Apply fisheye distortion correction using calibration parameters

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    calibration_file : str
        Path to fisheye calibration parameters

    Returns:
    --------
    np.ndarray
        Corrected image array

    Raises:
    ------
    ValueError
        If calibration file cannot be loaded
    """
    try:
        calibration_params = load_calibration(calibration_file)
        corrected_image = correct_distortion(image, calibration_params)
        logger.info("Fisheye distortion correction applied")
        return corrected_image
    except Exception as e:
        logger.error(f"Error in fisheye correction: {str(e)}")
        raise

def solve_plate(image_path: str, api_key: Optional[str] = None) -> WCS:
    """
    Use astrometry.net for plate solving

    Parameters:
    -----------
    image_path : str
        Path to the input image file
    api_key : str, optional
        Astrometry.net API key

    Returns:
    --------
    WCS
        World Coordinate System object

    Raises:
    ------
    RuntimeError
        If plate solving fails
    """
    try:
        astrometry_net = AstrometryNet()
        if api_key:
            astrometry_net.api_key = api_key

        # Upload the image and get the job ID
        job_id = astrometry_net.upload(image_path)
        
        # Wait for the job to complete
        astrometry_net.wait_for_job(job_id)
        
        # Get the results
        result = astrometry_net.get_job_status(job_id)
        
        if result['status'] == 'success':
            # Get the WCS solution
            wcs = astrometry_net.get_wcs_solution(job_id)
            logger.info("Plate solving successful")
            return wcs
        else:
            raise RuntimeError(f"Plate solving failed: {result.get('error', 'unknown error')}")
    except Exception as e:
        logger.error(f"Error in plate solving: {str(e)}")
        raise

def convert_to_healpix(
    image: np.ndarray, wcs: WCS, healpix_nside: int = 256
) -> np.ndarray:
    """
    Convert image pixels to HEALPix coordinates

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    wcs : WCS
        World Coordinate System object
    healpix_nside : int, optional
        HEALPix resolution parameter

    Returns:
    --------
    np.ndarray
        HEALPix map of the image
    """
    try:
        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x = x.flatten()
        y = y.flatten()

        sky_coords = wcs.pixel_to_world(x, y)
        healpix_indices = hp.ang2pix(
            healpix_nside, sky_coords.ra.deg, sky_coords.dec.deg, nest=True
        )

        logger.info("HEALPix conversion completed")
        return healpix_indices.reshape((h, w))
    except Exception as e:
        logger.error(f"Error in HEALPix conversion: {str(e)}")
        raise

def process_image(
    image_path: str,
    healpix_nside: int = 256,
    calibration_file: Optional[str] = None,
    api_key: Optional[str] = None
) -> np.ndarray:
    """
    Complete processing pipeline for converting an image to HEALPix format

    Parameters:
    -----------
    image_path : str
        Path to the input image file
    healpix_nside : int, optional
        HEALPix resolution parameter
    calibration_file : str, optional
        Path to fisheye calibration parameters
    api_key : str, optional
        Astrometry.net API key

    Returns:
    --------
    np.ndarray
        HEALPix map of the image

    Raises:
    ------
    ValueError
        If image loading fails or image is not 16-bit
    RuntimeError
        If plate solving fails
    """
    try:
        # Load image
        image = load_image(image_path)
        
        # Correct fisheye distortion
        image = correct_fisheye_distortion(image, calibration_file)
        
        # Solve plate
        wcs = solve_plate(image_path, api_key)
        
        # Convert to HEALPix
        return convert_to_healpix(image, wcs, healpix_nside)
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise
