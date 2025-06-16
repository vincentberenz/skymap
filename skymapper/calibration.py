from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

from .logger import logger
from .patches import extract_patches_from_file
from .plate_solver import solve_plate


def detect_stars_and_coordinates(
    image_path: str,
    api_key: Optional[str],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect stars in an image and convert their coordinates to sky coordinates.
    Uses the plate solver's internal star detection for better accuracy.

    Parameters:
    -----------
    image_path : str
        Path to the input image
    api_key : str, optional
        Astrometry.net API key

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] or None
        Tuple containing (sky_coords, star_coords) if successful, None otherwise
    """
    try:
        # Solve plate and get both WCS and star coordinates
        result = solve_plate(
            image_path=image_path,
            api_key=api_key,
        )
        if result is None:
            logger.error(f"Plate solving failed for {image_path}")
            return None

        wcs, stars = result

        # Convert star coordinates to numpy arrays
        if len(stars) < min_stars:
            logger.warning(
                f"Found only {len(stars)} stars in {image_path} (minimum required: {min_stars})"
            )
            return None

        # Extract star coordinates
        star_coords = np.array([(star["x"], star["y"]) for star in stars])

        # Convert to sky coordinates
        sky_coords = np.array([wcs.all_pix2world(x, y, 0) for x, y in star_coords])  # type: ignore[attr-defined]

        logger.info(f"Detected {len(star_coords)} stars in {Path(image_path).name}")
        return sky_coords, star_coords

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None


def process_patch(
    patch: PatchInfo,
    api_key: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process a single image patch to detect stars and convert to sky coordinates
    using plate solving.

    Parameters:
    -----------
    patch : PatchInfo
        Patch information including image data and coordinates
    api_key : str
        Astrometry.net API key

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] or None
        Tuple containing (sky_coords, star_coords) for this patch if successful, None otherwise.
        star_coords are in the original image's coordinate system.
    """
    try:
        # Save patch to a temporary file for plate solving
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            patch_path = tmp_file.name
            
        try:
            # Save the patch image
            cv2.imwrite(patch_path, patch.image)
            
            # Solve plate for this patch
            wcs_result = solve_plate(
                image_path=patch_path,
                api_key=api_key,
            )
        finally:
            # Clean up the temporary file
            try:
                os.unlink(patch_path)
            except OSError:
                pass
        
        if wcs_result is None:
            logger.error(f"Failed to solve plate for patch at ({patch.x}, {patch.y})")
            return None

        wcs, star_coords = wcs_result

        if not star_coords:
            logger.warning(f"No stars detected in patch at ({patch.x}, {patch.y})")
            return None

        # Convert image coordinates to sky coordinates
        sky_coords = []
        for star in star_coords:
            # Convert star coordinates from patch-relative to original image coordinates
            x_img = star['x'] + patch.x
            y_img = star['y'] + patch.y
            
            # Convert to sky coordinates
            ra, dec = wcs.all_pix2world(x_img, y_img, 0)  # type: ignore[attr-defined]
            sky_coords.append((ra, dec))

        return np.array(sky_coords), np.array([[s['x'] + patch.x, s['y'] + patch.y] for s in star_coords])

    except Exception as e:
        logger.error(f"Error processing patch at ({patch.x}, {patch.y}): {e}")
        return None


def generate_patches(
    image_paths: List[str],
    patch_size: int = 512,
    patch_overlap: int = 256,
) -> List[PatchInfo]:
    """Generate image patches for processing.

    Args:
        image_paths: List of paths to input images
        patch_size: Size of each patch in pixels
        patch_overlap: Overlap between patches in pixels

    Returns:
        List of PatchInfo objects containing the patches
    """
    patches: List[PatchInfo] = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Could not read image {img_path}")
            continue

        img_patches = extract_patches_from_image(
            img=img,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            image_path=str(img_path),
        )
        patches.extend(img_patches)
    return patches


@dataclass
class Star:
    """Represents a detected star with its image and sky coordinates."""

    image_x: float  # x-coordinate in the image
    image_y: float  # y-coordinate in the image
    ra: float  # Right ascension in degrees
    dec: float  # Declination in degrees

    @property
    def sky_coords(self) -> np.ndarray:
        """Return sky coordinates as a numpy array [ra, dec]."""
        return np.array([self.ra, self.dec])

    @property
    def image_coords(self) -> np.ndarray:
        """Return image coordinates as a numpy array [x, y]."""
        return np.array([self.image_x, self.image_y])


@dataclass
class PatchInfo:
    """Class to store information about an image patch."""

    image: np.ndarray  # The actual image data as a numpy array
    x: int  # X coordinate in the original image
    y: int  # Y coordinate in the original image
    original_image_path: str = ""  # Path to the original image this patch is from
    stars: List[Star] = field(default_factory=list)

    def __post_init__(self):
        """Validate the image data."""
        if not isinstance(self.image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if len(self.image.shape) not in (2, 3):
            raise ValueError("Image must be 2D (grayscale) or 3D (color) array")

    @property
    def width(self) -> int:
        """Return the width of the patch."""
        return self.image.shape[1]

    @property
    def height(self) -> int:
        """Return the height of the patch."""
        return self.image.shape[0]

    def center(self) -> Tuple[float, float]:
        """Return the center coordinates of the patch in the original image."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def to_rect(self) -> Tuple[int, int, int, int]:
        """Return the patch as (x, y, width, height) tuple in the original image."""
        return (self.x, self.y, self.width, self.height)
        
    def save(self, path: str) -> None:
        """Save the patch to a file.
        
        Args:
            path: Path where to save the patch image
        """
        cv2.imwrite(path, self.image)


def detect_stars_in_patch(
    patch: PatchInfo, api_key: str, star_threshold: float = 0.8, min_stars: int = 5
) -> None:
    """
    Detect stars in a single patch and store them in the patch object.

    Args:
        patch: The patch to process (contains image data)
        api_key: Astrometry.net API key for plate solving
        star_threshold: Threshold for star detection (0-1)
        min_stars: Minimum number of stars required for successful detection
    """
    try:
        # Create a temporary file for the patch image
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            patch_path = tmp_file.name
        
        try:
            # Save the patch image
            cv2.imwrite(patch_path, patch.image)
            
            # Detect stars using plate solving
            result = detect_stars_and_coordinates(
                image_path=patch_path,
                api_key=api_key,
                star_threshold=star_threshold,
                min_stars=min_stars,
            )
        finally:
            # Clean up the temporary file
            try:
                os.unlink(patch_path)
            except OSError:
                pass
        
        if result is not None:
            sky_coords, star_coords = result

            # Create Star objects and add them to the patch
            for (ra, dec), (x, y) in zip(sky_coords, star_coords):
                # Adjust coordinates based on patch position in the original image
                abs_x = x + patch.x
                abs_y = y + patch.y
                patch.stars.append(Star(image_x=abs_x, image_y=abs_y, ra=ra, dec=dec))

            logger.info(
                f"Detected {len(patch.stars)} stars in patch at ({patch.x}, {patch.y})"
            )
        else:
            logger.warning(f"No stars detected in patch at ({patch.x}, {patch.y})")
            
    except Exception as e:
        logger.warning(f"Error detecting stars in patch at ({patch.x}, {patch.y}): {e}")


def calibrate_fisheye(
    stars: List[Star], image_size: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """
    Perform fisheye camera calibration using detected stars.

    Args:
        stars: List of detected stars with image and sky coordinates
        image_size: Tuple of (height, width) of the image

    Returns:
        Dictionary containing calibration parameters:
        - 'camera_matrix': 3x3 camera matrix
        - 'dist_coeffs': Distortion coefficients

    Raises:
        ValueError: If not enough stars are provided for calibration
    """
    if len(stars) < 10:  # Minimum number of stars needed for calibration
        raise ValueError(
            f"At least 10 stars are required for calibration, got {len(stars)}"
        )

    # Prepare data for calibration
    image_points = np.array([star.image_coords for star in stars], dtype=np.float32)
    sky_coords = np.array([(star.ra, star.dec) for star in stars])

    # Convert sky coordinates to 3D unit vectors
    ra_rad = np.radians(np.array([s.ra for s in stars]))
    dec_rad = np.radians(np.array([s.dec for s in stars]))

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    object_points = np.column_stack((x, y, z)).astype(np.float32)

    # Estimate camera matrix
    height, width = image_size
    focal = max(width, height) * 0.8  # Initial guess for focal length
    cx, cy = width / 2, height / 2
    camera_matrix = np.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float32
    )

    # Initialize distortion coefficients
    dist_coeffs = np.zeros(4, dtype=np.float32)

    # Perform calibration
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    try:
        # Initialize rotation and translation vectors
        # Convert to the format expected by OpenCV
        obj_pts = np.array([object_points], dtype=np.float32)
        img_pts = np.array([image_points], dtype=np.float32)

        # Initialize rvecs and tvecs with correct shapes
        rvecs = np.zeros((1, 1, 3), dtype=np.float64)
        tvecs = np.zeros((1, 1, 3), dtype=np.float64)

        # Perform the fisheye calibration
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            imageSize=(width, height),
            K=camera_matrix,
            D=dist_coeffs,
            rvecs=[rvecs],
            tvecs=[tvecs],
            flags=flags,
            criteria=criteria,
        )

        if not ret:
            raise RuntimeError("Fisheye calibration failed to converge")

        return {"camera_matrix": mtx, "dist_coeffs": dist.ravel(), "rms": ret}

    except cv2.error as e:
        logger.error("OpenCV error during fisheye calibration: %s", e)
        raise RuntimeError(f"Fisheye calibration failed: {e}")


def process_patches(
    patch_dir: str,
    api_key: str,
    output_file: str,
) -> Dict[str, np.ndarray]:
    """
    Process image patches to generate calibration parameters.
    First analyzes patches to select the best ones, then processes only those.

    Args:
        patch_dir: Directory containing image patches
        api_key: Astrometry.net API key for plate solving
        output_file: Path to save calibration file

    Returns:
        Dictionary containing calibration parameters
    """
    # Load and select the best patches
    all_patches = load_patches_from_directory(patch_dir)
    selected_patches = select_best_patches(all_patches)
    logger.info(f"Selected {len(selected_patches)} patches for calibration")

    # Detect stars in all selected patches
    for patch in selected_patches:
        detect_stars_in_patch(
            patch=patch,
            api_key=api_key,
            star_threshold=0.8,  # Default threshold
            min_stars=5,  # Minimum stars required
        )

    # Collect all stars from all patches
    all_stars = []
    for patch in selected_patches:
        all_stars.extend(patch.stars)

    if not all_stars:
        raise ValueError("No stars were detected in any of the selected patches")

    logger.info(
        f"Using {len(all_stars)} total stars from {len(selected_patches)} patches"
    )

    # Get image size from the first patch
    first_patch = selected_patches[0]
    height, width = cv2.imread(first_patch.path, cv2.IMREAD_GRAYSCALE).shape

    # Perform fisheye calibration
    calibration_result = calibrate_fisheye(all_stars, (height, width))

    # Save the calibration result
    if output_file:
        fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", calibration_result["camera_matrix"])
        fs.write("dist_coeffs", calibration_result["dist_coeffs"])
        fs.release()
        logger.info(f"Calibration parameters saved to {output_file}")

    return calibration_result


def load_patches_from_directory(patch_dir: str) -> List[PatchInfo]:
    """
    Load and parse patch files from a directory into PatchInfo objects.

    Args:
        patch_dir: Directory containing patch files

    Returns:
        List of PatchInfo objects for valid patches

    Raises:
        ValueError: If no valid patch files are found
    """
    # Get all patch files with common extensions
    patch_files = list(Path(patch_dir).glob("*.tiff")) + list(
        Path(patch_dir).glob("*.tif")
    )

    if not patch_files:
        raise ValueError("No patch files found in the specified directory")

    logger.info(f"Found {len(patch_files)} patches for processing...")

    # Create PatchInfo objects for all valid patches
    all_patches = []
    for patch_file in patch_files:
        try:
            # Extract position from filename: patch_X_Y_W_H_*.tiff
            parts = patch_file.stem.split("_")
            if len(parts) >= 5 and parts[0] == "patch":
                x, y, w, h = map(int, parts[1:5])
                all_patches.append(
                    PatchInfo(
                        path=str(patch_file),
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                    )
                )
        except Exception as e:
            logger.warning(f"Could not process patch {patch_file}: {e}")
            continue

    if not all_patches:
        raise ValueError("No valid patches found in the specified directory")

    return all_patches


def score_patch_quality(patch: PatchInfo) -> float:
    """Score a patch based on how likely it contains only stars (no clouds/objects).

    Higher scores indicate better quality star fields with these characteristics:
    - Few bright stars (avoids overexposed regions)
    - Uniform background (avoids clouds/light pollution gradients)
    - Good contrast between stars and background
    - Avoids large bright areas (clouds/objects)

    Args:
        patch: PatchInfo object containing the image path and dimensions

    Returns:
        float: Quality score between 0 (worst) and 1 (best)
    """
    try:
        # Read the image in grayscale
        img = cv2.imread(patch.path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0

        # Crop to patch dimensions if needed
        if (patch.width, patch.height) != img.shape[::-1]:
            img = img[patch.y : patch.y + patch.height, patch.x : patch.x + patch.width]

        if img.size == 0:
            return 0.0

        # Normalize image to 0-1 range
        img = img.astype(np.float32) / 255.0

        # 1. Check for uniform background (low std in low-pass filtered image)
        blur = cv2.GaussianBlur(img, (0, 0), 5)
        bg_std = np.std(blur)

        # 2. Check for good contrast (high dynamic range)
        contrast = np.percentile(img, 99) - np.percentile(img, 1)

        # 3. Check for star-like features (high frequency components)
        laplacian = cv2.Laplacian(img, cv2.CV_32F)
        high_freq_energy = np.var(laplacian)

        # 4. Check for large bright areas (clouds/objects)
        bright_area = np.mean(img > 0.8)  # Percentage of very bright pixels

        # 5. Check for too many small bright spots (noise)
        _, binary = cv2.threshold(img, 0.7, 1.0, cv2.THRESH_BINARY)
        num_components = ndimage.label(binary)[1]

        # Normalize and combine scores (weights can be adjusted)
        bg_score = np.exp(-bg_std * 10)  # Lower background variation is better
        contrast_score = np.tanh(contrast * 5)  # Higher contrast is better
        hf_score = np.tanh(high_freq_energy * 10)  # Some high frequency is good
        bright_area_penalty = 1.0 - min(
            bright_area * 2, 1.0
        )  # Penalize large bright areas
        component_penalty = 1.0 - min(
            num_components / 100.0, 1.0
        )  # Penalize too many components

        # Combine scores with weights
        score = (
            0.3 * bg_score
            + 0.3 * contrast_score
            + 0.2 * hf_score
            + 0.1 * bright_area_penalty
            + 0.1 * component_penalty
        )

        return float(np.clip(score, 0, 1))

    except Exception as e:
        logger.warning(f"Error scoring patch {patch.path}: {e}")
        return 0.0


def select_best_patches(
    patches: List[PatchInfo], max_patches: int = 6, max_workers: Optional[int] = None
) -> List[PatchInfo]:
    """
    Select the best patches for calibration based on quality and distribution.
    Always returns at least some patches as long as input patches are provided.

    Args:
        patches: List of PatchInfo objects
        max_patches: Maximum number of patches to select
        max_workers: Maximum number of worker processes to use for parallel scoring.
                    If None, uses the number of processors on the machine.
                    Set to 1 to disable parallel processing.

    Returns:
        List of selected PatchInfo objects, sorted by quality (best first)
    """
    if not patches:
        return []

    # For small number of patches, don't bother with parallel processing
    if len(patches) < 5 or max_workers == 1:
        for patch in patches:
            patch.score = score_patch_quality(patch)
    else:
        # Score patches in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scoring tasks
            future_to_patch = {
                executor.submit(score_patch_quality, patch): patch for patch in patches
            }

            # Collect results as they complete
            for future in as_completed(future_to_patch):
                patch = future_to_patch[future]
                try:
                    patch.score = future.result()
                except Exception as e:
                    logger.warning(f"Error scoring patch {patch.path}: {e}")
                    patch.score = 0.0

    # Sort by score (descending)
    sorted_patches = sorted(patches, key=lambda p: p.score, reverse=True)

    # Log top scores for debugging
    top_scores = [
        f"{p.score:.2f}" for p in sorted_patches[: min(5, len(sorted_patches))]
    ]
    logger.debug(f"Top patch scores: {', '.join(top_scores)}")

    # Return up to max_patches best patches
    return sorted_patches[:max_patches]


def generate_patches_from_image(
    image_path: str, patch_size: int = 1024, patch_overlap: float = 0.5
) -> List[PatchInfo]:
    """
    Generate PatchInfo instances from an input image.

    Args:
        image_path: Path to the input image
        patch_size: Size of the square patches to extract
        patch_overlap: Overlap factor between patches (0-1)
    Returns:
        List of PatchInfo instances with image data and original_image_path set
    Raises:
        ValueError: If the image cannot be loaded or processed
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load and process the image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Generate patches for this image
    patches = []
    overlap_px = int(patch_size * patch_overlap)
    
    # Calculate number of patches in each dimension
    h, w = img.shape[:2]
    x_steps = (w - patch_size) // (patch_size - overlap_px) + 1
    y_steps = (h - patch_size) // (patch_size - overlap_px) + 1
    
    for y in range(y_steps):
        for x in range(x_steps):
            # Calculate patch coordinates
            x0 = x * (patch_size - overlap_px)
            y0 = y * (patch_size - overlap_px)
            x1 = min(x0 + patch_size, w)
            y1 = min(y0 + patch_size, h)
            
            # Extract the patch
            patch_img = img[y0:y1, x0:x1]
            
            # Create PatchInfo
            patch = PatchInfo(
                image=patch_img,
                x=x0,
                y=y0,
                original_image_path=str(img_path)
            )
            patches.append(patch)
    
    return patches


def detect_stars_in_patches(
    patches: List[PatchInfo],
    api_key: str,
    star_threshold: float = 0.8,
    min_stars: int = 5,
) -> List[Star]:
    """
    Detect stars in a list of patches.

    Args:
        patches: List of PatchInfo instances
        api_key: API key for astrometry.net
        star_threshold: Threshold for star detection (0-1)
        min_stars: Minimum number of stars required per patch

    Returns:
        List of all detected stars

    Raises:
        ValueError: If no stars are detected in any patch
    """
    all_stars = []

    for patch in patches:
        detect_stars_in_patch(
            patch=patch,
            api_key=api_key,
            star_threshold=star_threshold,
            min_stars=min_stars,
        )
        all_stars.extend(patch.stars)

    if not all_stars:
        raise ValueError("No stars were detected in any of the patches")

    logger.info(f"Detected {len(all_stars)} total stars from {len(patches)} patches")
    return all_stars


def generate_patches_from_images(
    image_paths: Union[str, List[str]],
    patch_size: int = 1024,
    patch_overlap: float = 0.5,
) -> List[PatchInfo]:
    """
    Generate patches from a list of input images.

    Args:
        image_paths: Single path or list of paths to input images
        patch_size: Size of the square patches to extract
        patch_overlap: Overlap factor between patches (0-1)

    Returns:
        List of PatchInfo instances

    Raises:
        ValueError: If no valid images are found or no patches can be generated
    """
    # Convert single path to list if needed
    if isinstance(image_paths, str):
        image_paths_list = [image_paths]
    else:
        image_paths_list = image_paths

    if not image_paths_list:
        raise ValueError("No image paths provided")

    logger.info(f"Processing {len(image_paths_list)} input images")

    # Generate patches from all images
    all_patches = []
    for img_path in image_paths_list:
        try:
            patches = generate_patches_from_image(
                image_path=img_path, patch_size=patch_size, patch_overlap=patch_overlap
            )
            all_patches.extend(patches)
            logger.debug(f"Generated {len(patches)} patches from {img_path}")
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            continue

    if not all_patches:
        raise ValueError("No valid patches could be generated from the input images")

    logger.info(
        f"Generated {len(all_patches)} patches from {len(image_paths_list)} images"
    )
    return all_patches


def visualize_selected_patches(
    selected_patches: List[PatchInfo], max_display_size: int = 1920
) -> bool:
    """
    Visualize selected patches on original images and prompt user to continue or quit.

    Args:
        selected_patches: List of selected PatchInfo objects
        max_display_size: Maximum size for display (longest side will be resized to this)

    Returns:
        bool: True if user wants to continue, False if they want to quit
    """
    try:
        from pathlib import Path

        import cv2

        # Group patches by original image
        patches_by_image = {}
        for patch in selected_patches:
            if (
                not hasattr(patch, "original_image_path")
                or not patch.original_image_path
            ):
                continue

            img_path = Path(patch.original_image_path)
            if img_path not in patches_by_image:
                patches_by_image[img_path] = []
            patches_by_image[img_path].append(patch)

        if not patches_by_image:
            logger.warning("No original image paths found in patches")
            return True

        for img_path, patches in patches_by_image.items():
            # Load the original image in color for visualization
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not load image for visualization: {img_path}")
                continue

            # Create a copy for drawing
            vis_img = img.copy()

            # Draw all patches on the image
            for i, patch in enumerate(patches):
                # Draw rectangle for the patch
                x, y, w, h = patch.x, patch.y, patch.width, patch.height
                color = (0, 255, 0)  # Green
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)

                # Add patch number
                cv2.putText(
                    vis_img,
                    str(i + 1),
                    (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            # Resize for display if needed
            h, w = vis_img.shape[:2]
            if max(h, w) > max_display_size:
                scale = max_display_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                vis_img = cv2.resize(vis_img, new_size, interpolation=cv2.INTER_AREA)

            # Show the image
            cv2.imshow(
                "Selected Patches (press any key to continue, q to quit)", vis_img
            )

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                cv2.destroyAllWindows()
                return False

        cv2.destroyAllWindows()
        return True

    except Exception as e:
        logger.error(f"Error in patch visualization: {e}")
        return True  # Continue anyway if visualization fails


def generate_calibration_from_stars(
    image_paths: Union[str, List[str]],
    api_key: str,
    output_file: str = "calibration.xml",
    patch_size: int = 1024,
    patch_overlap: float = 0.5,
    interactive: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate fisheye camera calibration parameters using star positions from night sky images.

    The function works by:
    1. Creating patches from input images
    2. Analyzing patches to select the most promising ones
    3. Using only the selected patches for calibration

    Parameters:
    -----------
    image_paths : Union[str, List[str]]
        Single path or list of paths to night sky images
    api_key : str
        API key for astrometry.net
    output_file : str, optional
        Path to save the calibration file
    patch_size : int, optional
        Size of the square patches to extract
    patch_overlap : float, optional
        Overlap factor between patches (0-1)
    interactive : bool, optional
        If True, displays the selected patches and prompts the user to continue or quit

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing calibration parameters:
        - 'camera_matrix': 3x3 camera matrix
        - 'dist_coeffs': Distortion coefficients
        - 'rms': Root mean square reprojection error

    Raises:
    -------
    ValueError
        If no valid patches can be generated or no stars are detected
    """
    logger.info("Starting fisheye camera calibration")

    # Generate patches from all input images
    all_patches = generate_patches_from_images(
        image_paths=image_paths, patch_size=patch_size, patch_overlap=patch_overlap
    )

    # Select best patches
    selected_patches = select_best_patches(all_patches)
    logger.info(f"Selected {len(selected_patches)} patches for calibration")

    # Show interactive visualization if requested
    if interactive:
        logger.info(
            "Displaying selected patches. Press any key to continue, 'q' to quit."
        )
        if not visualize_selected_patches(selected_patches):
            logger.info("Calibration cancelled by user")
            return {}

    # Detect stars in all selected patches
    all_stars = detect_stars_in_patches(
        patches=selected_patches, api_key=api_key, star_threshold=0.8, min_stars=5
    )

    # Get image size from the first patch
    first_patch = selected_patches[0]
    height, width = first_patch.height, first_patch.width

    try:
        # Perform fisheye calibration
        calibration_result = calibrate_fisheye(all_stars, (height, width))

        # Save the calibration result if output file is specified
        if output_file:
            fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
            fs.write("camera_matrix", calibration_result["camera_matrix"])
            fs.write("dist_coeffs", calibration_result["dist_coeffs"])
            fs.release()
            logger.info(f"Calibration parameters saved to {output_file}")

        return calibration_result

    except Exception as e:
        logger.error(f"Error in calibration: {str(e)}")
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
