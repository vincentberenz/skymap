from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from astropy.wcs import WCS

from .logger import logger
from .plate_solver import solve_plate


def extract_image_patches(
    image: np.ndarray,
    patch_size: int = 512,
    overlap: float = 0.5,
    min_stars: int = 5,
    star_threshold: float = 100,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Extract patches from an image for star detection

    Parameters:
    -----------
    image : np.ndarray
        Input image
    patch_size : int
        Size of each patch
    overlap : float
        Overlap ratio between patches (0.0 to 1.0)
    min_stars : int
        Minimum stars required in a patch
    star_threshold : float
        Threshold for star detection

    Returns:
    --------
    List of tuples containing (patch, x_offset, y_offset)
    """
    patches = []
    h, w = image.shape
    step = int(patch_size * (1 - overlap))

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Calculate patch bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + patch_size)
            y2 = min(h, y + patch_size)

            # Extract patch
            patch = image[y1:y2, x1:x2]

            # Detect stars in patch
            _, binary = cv2.threshold(patch, star_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            # Count stars
            star_count = sum(1 for contour in contours if cv2.contourArea(contour) > 10)

            if star_count >= min_stars:
                patches.append((patch, x1, y1))

    return patches


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

        # Extract patches from image
        patches = extract_image_patches(
            img,
            patch_size=512,
            overlap=0.5,
            min_stars=min_stars // 2,  # Lower threshold for patches
            star_threshold=star_threshold,
        )

        if not patches:
            logger.warning(f"No suitable patches found in {image_path}")
            return None

        # Process each patch
        all_sky_coords: list[tuple[float, float]] = []
        all_star_coords: list[tuple[float, float]] = []

        for patch, x_offset, y_offset in patches:
            result = process_patch(
                patch,
                x_offset,
                y_offset,
                image_path,
                api_key,
                star_threshold,
                min_stars,
            )
            if result is not None:
                sky_coords, star_coords = result
                all_sky_coords.extend(sky_coords)
                all_star_coords.extend(star_coords)

        if len(all_star_coords) < min_stars:
            logger.warning(f"Not enough stars found across patches in {image_path}")
            return None

        logger.info(
            f"Found {len(all_star_coords)} stars across patches in {image_path}"
        )
        return np.array(all_sky_coords), np.array(all_star_coords)

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
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
        wcs = solve_plate(image_path, api_key)

        # Convert image coordinates to sky coordinates
        sky_coords = []
        for x, y in star_coords:
            ra, dec = wcs.all_pix2world(x, y, 0)
            sky_coords.append((ra, dec))

        return np.array(sky_coords), star_coords

    except Exception as e:
        logger.error(f"Error processing patch: {str(e)}")
        return None


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

        # Extract patches from image
        patches = extract_image_patches(
            img,
            patch_size=512,
            overlap=0.5,
            min_stars=min_stars // 2,  # Lower threshold for patches
            star_threshold=star_threshold,
        )

        if not patches:
            logger.warning(f"No suitable patches found in {image_path}")
            return None

        # Process each patch
        all_sky_coords: list[tuple[float, float]] = []
        all_star_coords: list[tuple[float, float]] = []

        for patch, x_offset, y_offset in patches:
            result = process_patch(
                patch,
                x_offset,
                y_offset,
                image_path,
                api_key,
                star_threshold,
                min_stars,
            )
            if result is not None:
                sky_coords, star_coords = result
                all_sky_coords.extend(sky_coords)
                all_star_coords.extend(star_coords)

        if len(all_star_coords) < min_stars:
            logger.warning(f"Not enough stars found across patches in {image_path}")
            return None

        logger.info(
            f"Found {len(all_star_coords)} stars across patches in {image_path}"
        )
        return np.array(all_sky_coords), np.array(all_star_coords)

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None


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
    output_file : str, optional
        Path to save the calibration parameters
    min_stars : int, optional
        Minimum number of stars required per patch
    star_threshold : float, optional
        Threshold for star detection (higher value = fewer stars)
    patch_size : int, optional
        Size of image patches
    patch_overlap : float, optional
        Overlap ratio between patches (0.0 to 1.0)

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing calibration parameters
        'camera_matrix': Camera matrix as numpy array
        'dist_coeffs': Distortion coefficients as numpy array

    Raises:
    ------
    ValueError
        If not enough stars are found across all patches
    """
    try:
        # Initialize storage for points
        objpoints = []  # 3d points in sky coordinates
        imgpoints = []  # 2d points in image coordinates
        patches = []  # List of all patches across all images

        # Extract patches from all images first
        for image_path in image_paths:
            logger.info(f"Extracting patches from image: {image_path}")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not load image: {image_path}")
                continue

            # Extract patches
            image_patches = extract_image_patches(
                img,
                patch_size=patch_size,
                overlap=patch_overlap,
                min_stars=min_stars // 2,  # Lower threshold for patches
                star_threshold=star_threshold,
            )
            patches.extend(image_patches)

        if not patches:
            raise ValueError("No suitable patches found in any image")

        logger.info(f"Found {len(patches)} patches across all images")

        # Process each patch
        for patch, x_offset, y_offset in patches:
            logger.info(f"Processing patch at ({x_offset}, {y_offset})")
            result = process_patch(
                patch,
                x_offset,
                y_offset,
                image_paths[0],  # Use first image path for plate solving
                api_key,
                star_threshold,
                min_stars,
            )
            if result is not None:
                sky_coords, star_coords = result
                objpoints.append(sky_coords)
                imgpoints.append(star_coords)

        # Check if we have enough points
        total_stars = sum(len(points) for points in imgpoints)
        if total_stars < min_stars * 3:  # At least 3 patches with min_stars
            raise ValueError("Not enough stars found across all patches")

        # Get image size from any image
        img = cv2.imread(image_paths[0])
        h, w = img.shape[:2]

        # Initialize camera matrix with reasonable defaults
        K = np.zeros((3, 3))
        K[0, 0] = w  # focal length in pixels
        K[1, 1] = w  # same focal length for y
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
