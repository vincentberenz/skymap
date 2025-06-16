from typing import Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.nddata import Cutout2D
from astropy.time import Time
from astropy.wcs import WCS
from twirl import compute_wcs, find_peaks, gaia_radecs

IMAGE_PATH: str = (
    "/home/vberenz/Workspaces/skymap/test-twirl/nightskycam5_2025_04_30_23_42_30.tiff"
)


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


def load_and_process_fisheye_image(image_path: str) -> np.ndarray:
    """
    Load the TIFF image, convert to grayscale if needed, and extract central square region

    Args:
        image_path: Path to the TIFF image file

    Returns:
        Processed 2D grayscale image data as numpy array (uint8)
    """
    # Load the image using imageio
    data: np.ndarray = imageio.imread(image_path)
    print(f"Loaded image with shape: {data.shape}, dtype: {data.dtype}")

    # Convert to grayscale if image is color (3D array)
    if len(data.shape) == 3:
        # Use mean across color channels (simple grayscale conversion)
        data = data.mean(axis=2)

    # Ensure we have a 2D array
    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D array, got {len(data.shape)}D array")

    # Normalize to uint8
    data = normalize_to_uint8(data)

    # Extract central square patch (up to 1000x1000 pixels)
    size: int = min(1000, min(data.shape) // 2)  # Ensure we don't exceed image bounds
    center: Tuple[int, int] = (data.shape[0] // 2, data.shape[1] // 2)

    try:
        patch: Cutout2D = Cutout2D(data, center, size, mode="trim")
        print(f"Extracted patch with shape: {patch.data.shape}")
        return patch.data
    except Exception as e:
        print(f"Error creating cutout: {e}")
        print(f"Image shape: {data.shape}, center: {center}, size: {size}")
        raise


def configure_observation() -> Tuple[EarthLocation, Time, SkyCoord]:
    """
    Set up observation parameters for Tübingen

    Returns:
        Tuple of (location, observation_time, sky_coordinates)
    """
    # Define location (Tübingen, Germany)
    # Coordinates for Tübingen: 48.5216° N, 9.0576° E, 341m elevation
    location: EarthLocation = EarthLocation(
        lat=48.5216 * u.deg, lon=9.0576 * u.deg, height=341 * u.meter
    )

    # Set observation time
    obs_time: Time = Time("2025-04-30 23:42:00", scale="utc")

    # Calculate approximate pointing center
    # Note: For fisheye lens, assume zenith pointing
    alt: u.Quantity = 90 * u.degree  # Zenith
    az: u.Quantity = 0 * u.degree  # North

    # Create AltAz frame for the observation
    altaz_frame = AltAz(obstime=obs_time, location=location)

    # Create SkyCoord in AltAz frame and transform to ICRS
    sky_coords: SkyCoord = SkyCoord(alt=alt, az=az, frame=altaz_frame).icrs

    print(f"Observation location: {location}")
    print(f"Observation time: {obs_time.isot}")
    print(
        f"Pointing center (ICRS): RA={sky_coords.ra.deg:.4f}°, Dec={sky_coords.dec.deg:.4f}°"
    )

    return location, obs_time, sky_coords


def solve_plate(
    image_data: np.ndarray, center_coords: SkyCoord, fov_estimate: u.Quantity
) -> Optional[WCS]:
    """
    Perform plate solving with fisheye considerations and validate the solution

    Args:
        image_data: Processed image data
        center_coords: Estimated center coordinates
        fov_estimate: Estimated field of view

    Returns:
        WCS solution or None if solving fails
    """
    # Detect stars (more relaxed threshold due to fisheye distortion)
    stars_xy = find_peaks(image_data, threshold=5.0)[0:50]

    # Query GAIA stars (larger FOV due to fisheye distortion)
    gaia_stars = gaia_radecs(center_coords, 1.5 * fov_estimate)

    # Compute WCS with higher tolerance for fisheye distortion
    try:
        wcs = compute_wcs(stars_xy, gaia_stars[0:50], tolerance=15)

        # Print informative information about the WCS solution
        print("\nWCS Solution Information:")
        print("-" * 50)

        # Print basic WCS properties
        print(f"Number of matched stars: {len(stars_xy)}")
        print(f"Reference catalog: GAIA")
        print(f"Field of view: {fov_estimate}")

        # Calculate and print pixel scales
        pixel_scales = wcs.proj_plane_pixel_scales()
        print("\nPixel Scales:")
        print(f"X-axis: {pixel_scales[0]:.3f} degrees/pixel")
        print(f"Y-axis: {pixel_scales[1]:.3f} degrees/pixel")

        # Calculate pixel area
        pixel_area = wcs.proj_plane_pixel_area()
        print(f"Pixel area: {pixel_area:.3e} square degrees")

        # Validate coordinate transformations
        test_points = np.array(
            [
                [image_data.shape[0] // 4, image_data.shape[1] // 4],
                [image_data.shape[0] // 2, image_data.shape[1] // 2],
                [3 * image_data.shape[0] // 4, 3 * image_data.shape[1] // 4],
            ]
        )

        # Test forward and backward transformations
        print("\nTransformation Validation:")
        for point in test_points:
            sky = wcs.pixel_to_world(point[0], point[1])
            back_xy = wcs.world_to_pixel(sky)
            error = np.sqrt(np.sum((point - back_xy) ** 2))
            print(f"Point {point}: Round-trip error = {error:.2f} pixels")

        # Check bounds
        print("\nBounds Check:")
        print(
            f"RA range: [{wcs.pixel_to_world(0, 0).ra.degree:.2f}, "
            f"{wcs.pixel_to_world(image_data.shape[0], 0).ra.degree:.2f}] degrees"
        )
        print(
            f"DEC range: [{wcs.pixel_to_world(0, 0).dec.degree:.2f}, "
            f"{wcs.pixel_to_world(0, image_data.shape[1]).dec.degree:.2f}] degrees"
        )

        return wcs

    except Exception as e:
        print(f"\nWCS Solution Failed:")
        print(f"Error: {str(e)}")
        return None


def main() -> None:
    """Main function to execute the plate solving process"""
    # Load and process image
    image_path: str = IMAGE_PATH
    image_data: np.ndarray = load_and_process_fisheye_image(image_path)

    # Configure observation
    location: EarthLocation
    obs_time: Time
    center_coords: SkyCoord
    location, obs_time, center_coords = configure_observation()

    # Estimate FOV (8mm fisheye ~ 180° diagonal)
    fov_estimate: u.Quantity = 3.0 * u.degree  # Conservative estimate for central patch

    # Solve plate
    wcs: Optional[WCS] = solve_plate(image_data, center_coords, fov_estimate)

    # Save WCS solution
    # ... (Add your preferred method to save/use the WCS)


if __name__ == "__main__":
    main()
