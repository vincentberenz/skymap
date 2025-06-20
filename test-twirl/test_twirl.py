import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from loguru import logger
from twirl import compute_wcs, find_peaks, gaia_radecs

# Constants
ROOT_DIR = Path(__file__).parent
IMAGE_PATH = ROOT_DIR / "nightskycam5_2025_04_30_23_42_30.tiff"
LOG_LEVEL = "DEBUG"
LOG_FORMAT = "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
LOG_FILE = Path(f"skymapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Image processing constants
PATCH_SIZE = 400  # Initial size of the central patch to extract in pixels
STAR_DETECTION_THRESHOLD = 10  # Threshold for star detection
MAX_STARS = 50  # Maximum number of stars to use for plate solving
FOV_MULTIPLIER = 1.2  # Reduced from 1.5 to get a smaller initial search radius
FISHEYE_FOV = 180.0 * u.deg  # Total FOV of the fisheye lens
TRANSFORM_TOLERANCE = 0.5  # Tolerance for WCS transformation (pixels)
MAX_GAIA_RADIUS = 10.0 * u.deg  # Reduced from 20.0 to prevent timeouts
MAX_GAIA_STARS = 500  # Maximum number of stars to retrieve from GAIA
GAIA_QUERY_TIMEOUT = 30  # Timeout for GAIA queries in seconds

# Location constants (Tübingen, Germany)
OBS_LATITUDE = 48.5216 * u.deg
OBS_LONGITUDE = 9.0576 * u.deg
OBS_ELEVATION = 341 * u.meter
OBS_TIME = "2025-04-30 23:42:00"

# Paths
ASTROMETRY_WCS_PATH = ROOT_DIR / "astrometry_wcs.fits"

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    colorize=True,
)

logger.info("Debug logging enabled in terminal")


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


def calculate_fov_estimate(
    image_shape: Tuple[int, int], patch_size: int
) -> Tuple[u.Quantity, int, float]:
    """
    Calculate the field of view estimate for the central patch of a fisheye image,
    automatically adjusting parameters to stay within GAIA's search radius limits.

    Args:
        image_shape: Shape of the full image (height, width) in pixels
        patch_size: Initial size of the central patch in pixels

    Returns:
        Tuple of (fov_estimate, adjusted_patch_size, adjusted_multiplier)
        where fov_estimate is the final FOV in degrees,
        adjusted_patch_size is the possibly reduced patch size, and
        adjusted_multiplier is the possibly reduced FOV multiplier
    """
    # Start with the initial values
    adjusted_patch_size = patch_size
    adjusted_multiplier = float(FOV_MULTIPLIER)

    # Calculate the diagonal of the full image
    full_diag = np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)

    # Calculate the initial patch diagonal and FOV estimate
    patch_diag = np.sqrt(2) * adjusted_patch_size
    fov_estimate = (patch_diag / full_diag) * FISHEYE_FOV

    # Add a small safety margin (5% instead of 10%)
    fov_estimate = 1.05 * fov_estimate

    # Calculate the search radius
    search_radius = fov_estimate * adjusted_multiplier / 2.0

    # If initial radius is too large, reduce it
    if search_radius > MAX_GAIA_RADIUS:
        # First try reducing the multiplier
        if adjusted_multiplier > 1.0:
            adjusted_multiplier = min(
                adjusted_multiplier, (MAX_GAIA_RADIUS * 2) / fov_estimate
            )
            search_radius = fov_estimate * adjusted_multiplier / 2.0

        # If still too large, reduce the patch size
        if search_radius > MAX_GAIA_RADIUS:
            # Calculate the maximum patch diagonal that would keep us within limits
            max_patch_diag = (2 * MAX_GAIA_RADIUS * full_diag) / (
                FISHEYE_FOV * adjusted_multiplier * 1.05
            )
            adjusted_patch_size = int(
                (max_patch_diag / np.sqrt(2)) * 0.9
            )  # 90% to be safe

            # Ensure we don't go below minimum patch size
            adjusted_patch_size = max(50, adjusted_patch_size)

            # Recalculate with adjusted patch size
            patch_diag = np.sqrt(2) * adjusted_patch_size
            fov_estimate = (patch_diag / full_diag) * FISHEYE_FOV * 1.05  # 5% margin
            search_radius = fov_estimate * adjusted_multiplier / 2.0

    logger.debug(
        f"Adjusted parameters - Patch size: {adjusted_patch_size}, "
        f"FOV multiplier: {adjusted_multiplier:.2f}, "
        f"FOV estimate: {fov_estimate:.2f}, "
        f"Search radius: {search_radius:.2f}"
    )

    return fov_estimate, adjusted_patch_size, adjusted_multiplier


def load_and_process_fisheye_image(
    image_path: Union[str, Path], patch_size: int = None
) -> np.ndarray:
    """
    Load the TIFF image, convert to grayscale if needed, and extract central square region

    Args:
        image_path: Path to the TIFF image file
        patch_size: Size of the central square patch to extract (in pixels)

    Returns:
        Processed 2D grayscale image data as numpy array (uint8)
    """
    if patch_size is None:
        patch_size = PATCH_SIZE
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

        # Save the patch as TIFF
        output_path = ROOT_DIR / "processed_patch.tiff"
        imageio.imsave(output_path, patch.data, format="tiff")
        logger.info(f"Saved processed patch to {output_path}")

        return patch.data

    except Exception as e:
        logger.exception("Error processing image")
        raise


def load_astrometry_wcs(wcs_path: str) -> WCS:
    """
    Load a WCS solution from a FITS file created by astrometry.net

    Args:
        wcs_path: Path to the FITS file containing the WCS

    Returns:
        WCS object
    """
    with warnings.catch_warnings():
        # Suppress FITSFixedWarning about 'datfix' and 'unitfix'
        warnings.simplefilter("ignore", FITSFixedWarning)
        with fits.open(wcs_path) as hdul:
            wcs = WCS(hdul[0].header)
    return wcs


def compare_wcs(
    wcs1: WCS, wcs2: WCS, image_shape: Tuple[int, int], n_points: int = 9
) -> Dict[str, Any]:
    """
    Compare two WCS solutions by checking coordinate transformations at multiple points.

    Args:
        wcs1: First WCS solution
        wcs2: Second WCS solution
        image_shape: Shape of the image (height, width)
        n_points: Number of points to sample in each dimension (n_points² total points)

    Returns:
        Dictionary containing comparison results
    """
    height, width = image_shape
    center_x, center_y = width / 2, height / 2

    # Get center coordinates from both WCS solutions
    ra1_center, dec1_center = wcs1.all_pix2world([[center_x, center_y]], 0)[0]
    ra2_center, dec2_center = wcs2.all_pix2world([[center_x, center_y]], 0)[0]

    # Calculate center separation in arcseconds
    center_sep = (
        np.sqrt(
            (ra2_center - ra1_center) ** 2 * np.cos(np.radians(dec1_center)) ** 2
            + (dec2_center - dec1_center) ** 2
        )
        * 3600
    )

    # Generate grid of points
    x = np.linspace(0, width - 1, n_points, dtype=int)
    y = np.linspace(0, height - 1, n_points, dtype=int)
    xx, yy = np.meshgrid(x, y)

    # Flatten coordinates
    coords = np.column_stack((xx.ravel(), yy.ravel()))

    # Convert to celestial coordinates
    ra1, dec1 = wcs1.all_pix2world(coords[:, 0], coords[:, 1], 0)
    ra2, dec2 = wcs2.all_pix2world(coords[:, 0], coords[:, 1], 0)

    # Calculate angular separation in arcseconds
    sep = (
        np.sqrt((ra2 - ra1) ** 2 * np.cos(np.radians(dec1)) ** 2 + (dec2 - dec1) ** 2)
        * 3600
    )

    # Format coordinates for display
    def format_coord(ra: float, dec: float) -> str:
        return f"RA={ra:.6f}°, Dec={dec:.6f}°"

    # Calculate statistics
    stats = {
        "center_sep_arcsec": float(center_sep),
        "center_wcs1": format_coord(ra1_center, dec1_center),
        "center_wcs2": format_coord(ra2_center, dec2_center),
        "mean_sep_arcsec": float(np.mean(sep)),
        "median_sep_arcsec": float(np.median(sep)),
        "max_sep_arcsec": float(np.max(sep)),
        "min_sep_arcsec": float(np.min(sep)),
        "std_sep_arcsec": float(np.std(sep)),
        "n_points": len(coords),
    }
    return stats


def configure_observation() -> Tuple[EarthLocation, Time, SkyCoord]:
    """
    Set up observation parameters for Tübingen

    Returns:
        Tuple of (location, observation_time, sky_coordinates)
    """
    logger.info("Configuring observation parameters")

    try:
        # Define location (Tübingen, Germany)
        # Coordinates for Tübingen: 48.5216° N, 9.0576° E, 341m elevation
        location: EarthLocation = EarthLocation(
            lat=OBS_LATITUDE, lon=OBS_LONGITUDE, height=OBS_ELEVATION
        )
        logger.debug(f"Set observation location: {location}")

        # Set observation time
        obs_time: Time = Time(OBS_TIME, scale="utc")
        logger.debug(f"Set observation time: {obs_time.isot}")

        # Calculate approximate pointing center
        # Note: For fisheye lens, assume zenith pointing
        alt: u.Quantity = 90 * u.degree  # Zenith
        az: u.Quantity = 0 * u.degree  # North
        logger.debug(f"Assuming zenith pointing (alt={alt}, az={az})")

        # Create AltAz frame for the observation
        altaz_frame = AltAz(obstime=obs_time, location=location)

        # Create SkyCoord in AltAz frame and transform to ICRS
        sky_coords: SkyCoord = SkyCoord(alt=alt, az=az, frame=altaz_frame).icrs

        logger.info(f"Observation location: {location}")
        logger.info(f"Observation time: {obs_time.isot}")
        logger.info(
            f"Pointing center (ICRS): RA={sky_coords.ra.deg:.4f}°, Dec={sky_coords.dec.deg:.4f}°"
        )

        return location, obs_time, sky_coords

    except Exception as e:
        logger.exception("Error configuring observation")
        raise


def solve_plate(
    image_data: np.ndarray,
    center_coords: SkyCoord,
    fov_estimate: u.Quantity,
    fov_multiplier: float = None,
) -> Optional[WCS]:
    """
    Solve the plate for the given image data.

    Args:
        image_data: 2D numpy array containing the image data
        center_coords: SkyCoord of the estimated center of the image
        fov_estimate: Estimated field of view of the image
        fov_multiplier: Multiplier for FOV when querying the GAIA catalog

    Returns:
        WCS object if successful, None otherwise
    """
    logger.info("Starting plate solving")
    logger.debug(f"Image shape: {image_data.shape}, FOV estimate: {fov_estimate}")

    # Use provided multiplier or default to global constant
    if fov_multiplier is None:
        fov_multiplier = FOV_MULTIPLIER

    try:
        # Detect stars in the image
        logger.debug("Detecting stars in image")
        stars_xy = find_peaks(image_data, threshold=STAR_DETECTION_THRESHOLD)[
            0:MAX_STARS
        ]
        logger.info(f"Detected {len(stars_xy)} stars in the image")

        # Calculate search radius (in degrees) and ensure it's within limits
        search_radius = min(fov_estimate * fov_multiplier / 2.0, MAX_GAIA_RADIUS)
        search_fov = search_radius * 2.0  # Convert radius to diameter for gaia_radecs

        logger.debug(
            f"Querying GAIA catalog around {center_coords} with FOV {search_fov:.2f}"
        )

        try:
            # Query GAIA catalog with limited number of stars
            gaia_stars = gaia_radecs(
                center_coords,
                search_fov,  # gaia_radecs expects FOV (diameter), not radius
                limit=MAX_GAIA_STARS,
                circular=True,  # Use circular search area
            )

            logger.info(f"Retrieved {len(gaia_stars)} reference stars from GAIA")

            if len(gaia_stars) == 0:
                raise ValueError(
                    "No stars found in GAIA catalog for the given coordinates and FOV"
                )

            if len(gaia_stars) == MAX_GAIA_STARS:
                logger.warning(
                    f"Reached maximum number of stars ({MAX_GAIA_STARS}) in GAIA query. "
                    f"Consider reducing the search FOV (current: {search_fov:.2f}°)"
                )
        except Exception as e:
            logger.error(f"Failed to query GAIA catalog: {str(e)}")
            raise

        # Compute WCS with higher tolerance for fisheye distortion
        logger.debug("Computing WCS solution")
        wcs = compute_wcs(
            stars_xy, gaia_stars[0:MAX_STARS], tolerance=TRANSFORM_TOLERANCE
        )

        # Log WCS solution information
        logger.success("WCS solution computed successfully")
        logger.info("WCS Solution Information:")
        logger.info(f"- Number of matched stars: {len(stars_xy)}")
        logger.info(f"- Reference catalog: GAIA")
        logger.info(f"- Field of view: {fov_estimate:.2f}°")

        try:
            # Calculate and log pixel scales with proper unit handling
            pixel_scales = wcs.proj_plane_pixel_scales()

            # Convert pixel scales to arcsec/pixel
            x_scale = (pixel_scales[0].to(u.deg) * 3600).value  # Convert to arcsec
            y_scale = (pixel_scales[1].to(u.deg) * 3600).value  # Convert to arcsec

            logger.info("Pixel Scales:")
            logger.info(f"  - X-axis: {x_scale:.3f} arcsec/pixel")
            logger.info(f"  - Y-axis: {y_scale:.3f} arcsec/pixel")

            # Calculate and log pixel area in arcsec²
            pixel_area = x_scale * y_scale  # Already in arcsec²
            logger.info(f"Pixel area: {pixel_area:.3f} arcsec²")

        except Exception as e:
            logger.warning(f"Could not compute pixel scales: {str(e)}")
            logger.warning("This might be due to distortion in the fisheye projection")

        # Validate coordinate transformations
        test_points = np.array(
            [
                [image_data.shape[0] // 4, image_data.shape[1] // 4],
                [image_data.shape[0] // 2, image_data.shape[1] // 2],
                [3 * image_data.shape[0] // 4, 3 * image_data.shape[1] // 4],
            ]
        )

        logger.debug("Validating coordinate transformations")
        transformation_errors = []
        for point in test_points:
            sky = wcs.pixel_to_world(point[0], point[1])
            back_xy = wcs.world_to_pixel(sky)
            error = np.sqrt(np.sum((point - back_xy) ** 2))
            transformation_errors.append(error)
            logger.debug(f"Point {point}: Round-trip error = {error:.2f} pixels")

        avg_error = np.mean(transformation_errors)
        logger.info(f"Average transformation error: {avg_error:.2f} pixels")

        # Log bounds
        ra_start = wcs.pixel_to_world(0, 0).ra.degree
        ra_end = wcs.pixel_to_world(image_data.shape[0], 0).ra.degree
        dec_start = wcs.pixel_to_world(0, 0).dec.degree
        dec_end = wcs.pixel_to_world(0, image_data.shape[1]).dec.degree

        logger.info("Image bounds:")
        logger.info(f"  - RA range:  [{ra_start:.2f}°, {ra_end:.2f}°]")
        logger.info(f"  - DEC range: [{dec_start:.2f}°, {dec_end:.2f}°]")

        return wcs

    except Exception as e:
        logger.error(f"WCS solution failed: {str(e)}", exc_info=True)
        return None


def main() -> None:
    """Main function to run the plate solving process and compare WCS solutions."""
    logger.info("Starting plate solving process")

    try:
        # Load and process the fisheye image
        logger.info(f"Processing image: {IMAGE_PATH}")

        # Load the astrometry.net WCS solution
        if ASTROMETRY_WCS_PATH.exists():
            logger.info(f"Loading astrometry.net WCS from {ASTROMETRY_WCS_PATH}")
            astrometry_wcs = load_astrometry_wcs(str(ASTROMETRY_WCS_PATH))
        else:
            logger.warning(f"Astrometry WCS file not found: {ASTROMETRY_WCS_PATH}")
            astrometry_wcs = None

        # First load the full image to get its shape for FOV calculation
        full_image = imageio.imread(IMAGE_PATH)
        if len(full_image.shape) == 3:  # Convert color to grayscale
            full_image = full_image.mean(axis=2)

        # Calculate FOV estimate and get adjusted parameters
        fov_estimate, adjusted_patch_size, fov_multiplier = calculate_fov_estimate(
            full_image.shape, PATCH_SIZE
        )

        logger.info(f"Using FOV estimate: {fov_estimate:.2f}")
        logger.info(f"Using patch size: {adjusted_patch_size} (original: {PATCH_SIZE})")
        logger.info(
            f"Using FOV multiplier: {fov_multiplier:.2f} (original: {FOV_MULTIPLIER})"
        )

        # Now process the image with the adjusted patch size
        image_data: np.ndarray = load_and_process_fisheye_image(
            IMAGE_PATH, patch_size=adjusted_patch_size
        )

        # Configure observation parameters
        location, observation_time, center_coords = configure_observation()

        # Solve the plate with the adjusted parameters
        logger.info("Starting plate solving")
        twirl_wcs: Optional[WCS] = solve_plate(
            image_data, center_coords, fov_estimate, fov_multiplier=fov_multiplier
        )

        if twirl_wcs is None:
            logger.error("Failed to solve plate with twirl")
            return

        logger.success("Plate solving completed successfully!")

        # Compare WCS solutions if astrometry WCS is available
        if astrometry_wcs is not None:
            logger.info("Comparing WCS solutions...")
            comparison = compare_wcs(
                twirl_wcs,
                astrometry_wcs,
                image_shape=image_data.shape,
                n_points=11,  # 11x11 grid of points
            )

            logger.info("\n=== WCS Comparison Results ===")
            logger.info("Center Coordinate Comparison:")
            logger.info(f"  Twirl WCS:    {comparison['center_wcs1']}")
            logger.info(f"  Astrometry WCS: {comparison['center_wcs2']}")
            logger.info(
                f"  Center separation: {comparison['center_sep_arcsec']:.3f} arcsec\n"
            )

            logger.info("Grid Point Separations (arcseconds):")
            logger.info(
                f"  Mean: {comparison['mean_sep_arcsec']:.3f} "
                + "\u00b1"
                + f" {comparison['std_sep_arcsec']:.3f}"
            )
            logger.info(f"  Median: {comparison['median_sep_arcsec']:.3f}")
            logger.info(
                f"  Range: {comparison['min_sep_arcsec']:.3f} - {comparison['max_sep_arcsec']:.3f}"
            )
            logger.info(f"  Points compared: {comparison['n_points']}")
            logger.info("=" * 30)

            # Log a warning if the mean separation is large
            if comparison["mean_sep_arcsec"] > 60:  # 1 arcminute
                logger.warning("Large separation detected between WCS solutions!")
        else:
            logger.info("No astrometry.net WCS available for comparison")

    except Exception as e:
        logger.exception("An error occurred during plate solving")
        raise
    finally:
        logger.info("Plate solving process completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("Unhandled exception in main", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
