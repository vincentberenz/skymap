import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from astropy.wcs import WCS

from .plate_solver import solve_plate

logger = logging.getLogger(__name__)


def generate_calibration_from_stars(
    image_paths: List[str],
    api_key: Optional[str] = None,
    output_file: str = "calibration.xml",
    min_stars: int = 20,
    star_threshold: float = 100,
) -> Dict[str, np.ndarray]:
    """
    Generate fisheye camera calibration parameters using star positions from night sky images

    Parameters:
    -----------
    image_paths : List[str]
        List of paths to night sky images
    api_key : str, optional
        Astrometry.net API key
    output_file : str, optional
        Path to save the calibration parameters
    min_stars : int, optional
        Minimum number of stars required per image
    star_threshold : float, optional
        Threshold for star detection (higher value = fewer stars)

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing calibration parameters
        'camera_matrix': Camera matrix as numpy array
        'dist_coeffs': Distortion coefficients as numpy array

    Raises:
    ------
    ValueError
        If not enough stars are found across all images
    """
    try:
        # Initialize storage for points
        objpoints = []  # 3d points in sky coordinates
        imgpoints = []  # 2d points in image coordinates

        # Process each image
        for image_path in image_paths:
            logger.info(f"Processing image: {image_path}")

            # Load and process image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not load image: {image_path}")
                continue

            # Detect stars using simple thresholding
            _, binary = cv2.threshold(img, star_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter and get star coordinates
            stars = []
            for contour in contours:
                if cv2.contourArea(contour) > 10:  # Ignore small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    stars.append((x + w / 2, y + h / 2))

            if len(stars) < min_stars:
                logger.warning(f"Not enough stars found in {image_path}")
                continue

            # Convert to numpy array
            star_coords = np.array(stars)

            # Solve plate for this image
            wcs = solve_plate(image_path, api_key)

            # Convert image coordinates to sky coordinates
            sky_coords = []
            for x, y in star_coords:
                ra, dec = wcs.all_pix2world(x, y, 0)
                sky_coords.append((ra, dec))

            # Add points to our collection
            objpoints.append(np.array(sky_coords))
            imgpoints.append(star_coords)

            logger.info(f"Found {len(stars)} stars in {image_path}")

        # Check if we have enough points
        total_stars = sum(len(points) for points in imgpoints)
        if total_stars < min_stars * 3:  # At least 3 images with min_stars
            raise ValueError("Not enough stars found across all images")

        # Get image size from any image
        img = cv2.imread(image_paths[0])
        h, w = img.shape[:2]

        # Initialize camera matrix with reasonable defaults
        K = np.zeros((3, 3))
        K[0, 0] = w  # focal length in pixels
        K[1, 1] = h
        K[0, 2] = w / 2  # principal point
        K[1, 2] = h / 2
        D = np.zeros((4, 1))  # distortion coefficients

        # Perform calibration
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            (w, h),
            K,
            D,
            None,
            None,
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_CHECK_COND
            + cv2.fisheye.CALIB_FIX_SKEW,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        )

        # Save calibration parameters
        calibration_params = {"camera_matrix": K, "dist_coeffs": D}

        # Save to file
        fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", K)
        fs.write("dist_coeffs", D)
        fs.release()
        logger.info(f"Saved calibration parameters to {output_file}")

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
