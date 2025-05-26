from .logger import logger
from typing import Optional
from pathlib import Path
from astropy.wcs import WCS
from astroquery.astrometry_net import AstrometryNet

def solve_plate(
    image_path: str,
    api_key: Optional[str] = None,
    max_retries: int = 30,
    retry_delay: float = 5.0,  # seconds
) -> WCS:
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

        # Load image
        image_path = Path(image_path)
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")

        # Submit image to Astrometry.net
        job = client.upload(image_path)
        if not job:
            raise ValueError("Failed to upload image to Astrometry.net")

        # Wait for job to complete
        job_id = job['jobid']
        logger.info(f"Submitted job {job_id} to Astrometry.net")

        # Poll for results
        for _ in range(max_retries):
            result = client.job_status(job_id)
            if result['status'] == 'success':
                break
            elif result['status'] == 'failure':
                raise ValueError(f"Plate solving failed: {result.get('errormessage', 'Unknown error')}")
            logger.info(f"Waiting for job {job_id} to complete...")
            time.sleep(retry_delay)
        else:
            raise ValueError("Plate solving timed out")

        # Get WCS solution
        wcs_data = client.job_wcs(job_id)
        if not wcs_data:
            raise ValueError("Failed to get WCS solution from Astrometry.net")

        # Create WCS object from solution
        wcs = WCS(wcs_data)
        logger.info(f"Successfully solved plate for {image_path}")
        return wcs

    except Exception as e:
        logger.error(f"Error in plate solving: {str(e)}")
        raise
