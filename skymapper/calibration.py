from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
from astropy.wcs import WCS

from .logger import logger
from .plate_solver import solve_plate
from .utils import load_image


def detect_stars_and_coordinates(
    image_path: str, api_key: Optional[str], star_threshold: float, min_stars: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect stars in an image and convert their coordinates to sky coordinates

    Parameters:
    -----------
    image_path : str
        Path to the input image
    api_key : str, optional
        Astrometry.net API key
    star_threshold : float
        Threshold for star detection
    min_stars : int
        Minimum number of stars required per image

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] or None
        Tuple containing (sky_coords, star_coords) if successful, None otherwise
    """
    try:
        # Load and process image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Could not load image: {image_path}")
            return None

        # Detect stars
        _, binary = cv2.threshold(img, star_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter stars by size
        stars = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Minimum size
                x, y, w, h = cv2.boundingRect(contour)
                stars.append((x + w / 2, y + h / 2))

        if len(stars) < min_stars:
            logger.warning(f"Found only {len(stars)} stars in {image_path}")
            return None

        # Convert to numpy arrays
        star_coords = np.array(stars)

        # Solve plate
        result = solve_plate(image_path, api_key)
        if result is None:
            logger.error(f"Plate solving failed for {image_path}")
            return None

        wcs, _ = result  # Unpack the WCS and star list

        # Convert to sky coordinates
        sky_coords = np.array([wcs.all_pix2world(x, y, 0) for x, y in star_coords])  # type: ignore[attr-defined]

        return sky_coords, star_coords

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None


def process_patch(
    patch: np.ndarray,
    x_offset: int,
    y_offset: int,
    image_path: str,
    api_key: Optional[str],
    star_threshold: float,
    min_stars: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process a single image patch to detect stars and convert to sky coordinates

    Parameters:
    -----------
    patch : np.ndarray
        Image patch
    x_offset : int
        X offset of the patch in the original image
    y_offset : int
        Y offset of the patch in the original image
    image_path : str
        Path to the original image
    api_key : str, optional
        Astrometry.net API key
    star_threshold : float
        Threshold for star detection
    min_stars : int
        Minimum number of stars required

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] or None
        Tuple containing (sky_coords, star_coords) for this patch if successful, None otherwise
    """
    try:
        # Detect stars in patch
        _, binary = cv2.threshold(patch, star_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and get star coordinates
        stars = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)
                # Adjust coordinates for full image
                stars.append((x + x_offset + w / 2, y + y_offset + h / 2))

        if len(stars) < min_stars:
            return None

        # Convert to numpy array
        star_coords = np.array(stars)

        # Solve plate for this patch
        wcs_result = solve_plate(image_path, api_key)
        if wcs_result is None:
            logger.error(f"Failed to solve plate for patch in {image_path}")
            return None

        wcs, _ = wcs_result

        # Convert image coordinates to sky coordinates
        sky_coords = []
        for x, y in star_coords:
            ra, dec = wcs.all_pix2world(x, y, 0)  # type: ignore[attr-defined]
            sky_coords.append((ra, dec))

        return np.array(sky_coords), star_coords

    except Exception as e:
        logger.error(f"Error processing patch: {str(e)}")
        return None


def generate_patches(
    image_paths: List[str],
    output_dir: str,
    patch_size: int = 512,
    patch_overlap: int = 256,
) -> None:
    """Generate and save image patches for processing.

    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save patches
        patch_size: Size of each patch in pixels
        patch_overlap: Overlap between patches in pixels
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for img_path in image_paths:
        extract_patches_from_file(img_path, patch_size, patch_overlap, output_dir)


def process_patches(
    patch_dir: str,
    api_key: Optional[str],
    output_file: str,
    min_stars: int = 20,
    star_threshold: float = 100,
) -> Dict[str, np.ndarray]:
    """Process image patches to generate calibration parameters.

    Args:
        patch_dir: Directory containing image patches
        api_key: Astrometry.net API key
        output_file: Path to save calibration file
        min_stars: Minimum stars required per patch
        star_threshold: Threshold for star detection

    Returns:
        Dictionary containing calibration parameters
    """
    # TODO: Implement actual patch processing
    # This is a placeholder implementation
    calibration_params: Dict[str, np.ndarray] = {
        "camera_matrix": np.eye(3),
        "dist_coeffs": np.zeros(4),
    }
    return calibration_params


def generate_calibration_from_stars(
    image_paths: List[str],
    api_key: Optional[str] = None,
    output_file: str = "calibration.xml",
    min_stars: int = 20,
    star_threshold: float = 100,
    patch_size: int = 512,
    patch_overlap: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Generate fisheye camera calibration parameters using star positions from night sky images

    Parameters:
    -----------
    image_paths : List[str]
        List of paths to night sky images
    api_key : str, optional
        Astrometry.net API key
    """
    try:
        # Generate patches
        patch_dir = "patches"
        generate_patches(
            image_paths,
            patch_dir,
            patch_size=patch_size,
            patch_overlap=int(patch_size * patch_overlap),
        )
        logger.info(f"Generated patches saved to {patch_dir}")

        # Process patches
        calibration_params = process_patches(
            patch_dir,
            api_key,
            output_file,
            min_stars=min_stars,
            star_threshold=star_threshold,
        )
        logger.info(f"Calibration parameters saved to {output_file}")

        return calibration_params
    except Exception as e:
        logger.error(f"Error generating calibration: {str(e)}")
        raise


def load_calibration(calibration_file: str) -> Dict[str, np.ndarray]:
    """
    Load calibration parameters from file

    Parameters:
    -----------
    calibration_file : str
        Path to the calibration parameters file

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing calibration parameters
        'camera_matrix': Camera matrix as numpy array
        'dist_coeffs': Distortion coefficients as numpy array

    Raises:
    ------
    ValueError
        If calibration file cannot be loaded
    """
    try:
        # Load calibration parameters using OpenCV
        fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)

        if not fs.isOpened():
            raise ValueError(f"Failed to open calibration file: {calibration_file}")

        calibration_params = {
            "camera_matrix": fs.getNode("camera_matrix").mat(),
            "dist_coeffs": fs.getNode("dist_coeffs").mat(),
        }
        fs.release()
        logger.info(
            f"Successfully loaded calibration parameters from {calibration_file}"
        )
        return calibration_params
    except Exception as e:
        logger.error(f"Error loading calibration: {str(e)}")
        raise


def correct_distortion(
    image: np.ndarray, calibration_params: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Apply fisheye distortion correction to an image

    Parameters:
    -----------
    image : np.ndarray
        Input image array
    calibration_params : Dict[str, np.ndarray]
        Dictionary containing calibration parameters
        'camera_matrix': Camera matrix as numpy array
        'dist_coeffs': Distortion coefficients as numpy array

    Returns:
    --------
    np.ndarray
        Corrected image array

    Raises:
    ----
    ValueError
        If calibration parameters are invalid
    """
    try:
        # Get image dimensions
        h, w = image.shape[:2]

        # Get optimal new camera matrix
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            calibration_params["camera_matrix"],
            calibration_params["dist_coeffs"],
            (w, h),
            np.eye(3),
            balance=1.0,
        )

        # Undistort the image
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            calibration_params["camera_matrix"],
            calibration_params["dist_coeffs"],
            np.eye(3),
            new_camera_matrix,
            (w, h),
            cv2.CV_16SC2,
        )

        corrected_image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        logger.info("Fisheye distortion correction applied")
        return corrected_image
    except Exception as e:
        logger.error(f"Error correcting distortion: {str(e)}")
        raise
