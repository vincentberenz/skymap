import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from astropy.io import fits
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet

from .logger import logger


def load_solution(wcs_path: str) -> Tuple[WCS, List[Dict[str, float]]]:
    """
    Load a WCS solution from a FITS file.

    Returns:
    --------
    Tuple[WCS, List[Dict[str, float]]]
        WCS object and list of star coordinates with 'x' and 'y' keys
    """
    try:
        with fits.open(wcs_path) as hdul:
            wcs = WCS(hdul[0].header)
            # Extract star coordinates from the FITS header if available
            stars = []
            if "NSTARS" in hdul[0].header:
                n_stars = hdul[0].header["NSTARS"]
                for i in range(1, n_stars + 1):
                    if f"X{i}" in hdul[0].header and f"Y{i}" in hdul[0].header:
                        stars.append(
                            {
                                "x": float(hdul[0].header[f"X{i}"]),
                                "y": float(hdul[0].header[f"Y{i}"]),
                            }
                        )
            return wcs, stars
    except Exception as e:
        logger.error(f"Failed to load WCS solution from {wcs_path}: {e}")
        raise


def solve_plate(
    image_path: str,
    api_key: Optional[str] = None,
    max_retries: int = 30,
    retry_delay: float = 5.0,
) -> Optional[Tuple[WCS, List[Dict[str, float]]]]:
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
    ValueError
        If plate solving fails
    """
    try:
        # Initialize Astrometry.net client
        client = AstrometryNet()
        if api_key:
            client.api_key = api_key

        # Convert to Path object
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise ValueError(f"Image file not found: {image_path}")

        # Check if solution exists
        solution_path = image_path_obj.with_suffix(".wcs")
        if solution_path.exists():
            return load_solution(str(solution_path))

        # Submit image to Astrometry.net
        job = client.upload(image_path)
        if not job:
            raise ValueError("Failed to upload image to Astrometry.net")

        # Wait for job to complete
        job_id = job["jobid"]
        logger.info(f"Submitted job {job_id} to Astrometry.net")

        # Poll for results
        for _ in range(max_retries):
            result = client.job_status(job_id)
            if result["status"] == "success":
                break
            elif result["status"] == "failure":
                raise ValueError(
                    f"Plate solving failed: {result.get('errormessage', 'Unknown error')}"
                )
            logger.info(f"Waiting for job {job_id} to complete...")
            time.sleep(retry_delay)
        else:
            raise ValueError("Plate solving timed out")

        # Get WCS solution
        wcs_data = client.job_wcs(job_id)
        if not wcs_data:
            raise ValueError("Failed to get WCS solution from Astrometry.net")

        # Get star coordinates from the solution
        stars_info = client.job_stars(job_id)
        stars = []
        if stars_info:
            for star in stars_info:
                if "x" in star and "y" in star:
                    stars.append({"x": float(star["x"]), "y": float(star["y"])})

        # Create WCS object from solution
        wcs = WCS(wcs_data)

        # Save the solution with star coordinates
        solution_path = Path(image_path).with_suffix(".wcs")
        with fits.open(wcs_data) as hdul:
            hdr = hdul[0].header
            # Add star coordinates to header
            hdr["NSTARS"] = len(stars)
            for i, star in enumerate(stars, 1):
                hdr[f"X{i}"] = star["x"]
                hdr[f"Y{i}"] = star["y"]
            hdul.writeto(solution_path, overwrite=True)

        logger.info(
            f"Successfully solved plate for {image_path} with {len(stars)} stars"
        )
        return wcs, stars

    except Exception as e:
        logger.error(f"Error in plate solving: {str(e)}")
        raise
