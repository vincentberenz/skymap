import logging
from typing import List, Optional

import click
import numpy as np

from .calibration import generate_calibration_from_stars
from .mapper import (
    convert_to_healpix,
    correct_fisheye_distortion,
    load_image,
    process_image,
)
from .plate_solver import solve_plate
from .utils import plot_healpix_projection, visualize_healpix_map

logger = logging.getLogger(__name__)


def process_tiff_image(
    image_path: str,
    healpix_nside: int = 256,
    calibration_file: Optional[str] = None,
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
    show_plot: bool = True,
) -> np.ndarray:
    """
    Process a TIFF image through the HEALPix pipeline

    Parameters:
    -----------
    image_path : str
        Path to the input TIFF image file
    healpix_nside : int, optional
        HEALPix resolution parameter
    calibration_file : str, optional
        Path to fisheye calibration parameters
    api_key : str, optional
        Astrometry.net API key
    output_path : str, optional
        Path to save the visualization
    show_plot : bool, optional
        Whether to show the plot interactively

    Returns:
    --------
    np.ndarray
        HEALPix map of the image
    """
    try:
        logger.info(f"Processing image: {image_path}")

        # Run the processing pipeline
        healpix_map = process_image(
            image_path=image_path,
            healpix_nside=healpix_nside,
            calibration_file=calibration_file,
            api_key=api_key,
        )

        # Visualize the result
        if show_plot or output_path:
            visualize_healpix_map(healpix_map, output_path)

        return healpix_map
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


def process_image_cli(
    image_path: str,
    calibration_file: str,
    output: str,
    api_key: Optional[str] = None,
) -> None:
    """
    Process an image through the full pipeline

    Args:
        image_path: Path to the input image
        calibration_file: Path to fisheye calibration parameters
        output: Output file path
        api_key: Optional Astrometry.net API key
    """
    try:
        # Load and process the image
        image = load_image(image_path)
        image = correct_fisheye_distortion(image, calibration_file)
        wcs = solve_plate(image_path, api_key)
        healpix_map = convert_to_healpix(image, wcs)

        # Save the result
        np.save(output, healpix_map)
        logger.info(f"Successfully processed image and saved to {output}")
        click.echo(f"Successfully processed image and saved to {output}")

    except Exception as e:
        logger.error(str(e))
        click.echo(f"Error: {str(e)}", err=True)
        raise


def calibrate_cli(
    image_paths: List[str],
    api_key: Optional[str],
    output: str,
    min_stars: int,
    star_threshold: float,
) -> None:
    """
    Generate fisheye camera calibration parameters from night sky images.

    Args:
        image_paths: List of paths to night sky images
        api_key: Optional Astrometry.net API key
        output: Output file for calibration parameters
        min_stars: Minimum number of stars required per image
        star_threshold: Threshold for star detection
    """
    try:
        calibration_params = generate_calibration_from_stars(
            image_paths=image_paths,
            api_key=api_key,
            output_file=output,
            min_stars=min_stars,
            star_threshold=star_threshold,
        )
        click.echo(
            f"Successfully generated calibration parameters and saved to {output}"
        )
    except Exception as e:
        logger.error(str(e))
        click.echo(f"Error: {str(e)}", err=True)
        raise


@click.group()
def main():
    """Process night sky images and generate calibration parameters."""
    pass


@main.command()
@click.argument("image-path", type=click.Path(exists=True))
@click.option("-n", "--nside", default=256, help="HEALPix resolution parameter")
@click.option("-c", "--calibration", help="Path to fisheye calibration parameters")
@click.option("-a", "--api-key", help="Astrometry.net API key")
@click.option("-o", "--output", help="Path to save the visualization")
@click.option("--no-show", is_flag=True, help="Do not show interactive plot")
def visualize(
    image_path: str,
    nside: int,
    calibration: str,
    api_key: str,
    output: str,
    no_show: bool,
) -> None:
    """
    Process a TIFF image and visualize the HEALPix map.

    IMAGE_PATH: Path to the input TIFF image
    """
    try:
        # Process the image
        image = load_image(image_path)
        if calibration:
            image = correct_fisheye_distortion(image, calibration)
        wcs = solve_plate(image_path, api_key)
        healpix_map = convert_to_healpix(image, wcs, nside=nside)

        # Visualize the result
        visualize_healpix_map(healpix_map, show=not no_show)
        if output:
            np.save(output, healpix_map)
            logger.info(f"Successfully saved visualization to {output}")

        click.echo("Successfully processed and visualized image")

    except Exception as e:
        logger.error(str(e))
        click.echo(f"Error: {str(e)}", err=True)
        raise


@main.command()
@click.argument("image-path", type=click.Path(exists=True))
@click.option(
    "--calibration-file", required=True, help="Path to fisheye calibration parameters"
)
@click.option("--output", default="output.npy", help="Output file path")
@click.option("--api-key", help="Astrometry.net API key")
def process(
    image_path: str,
    calibration_file: str,
    output: str,
    api_key: Optional[str] = None,
) -> None:
    """
    Process a night sky image and convert it to HEALPix format.

    IMAGE_PATH: Path to the input image
    """
    process_image_cli(image_path, calibration_file, output, api_key)


@main.command("calibrate")
@click.argument("image-paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--api-key", help="Astrometry.net API key")
@click.option(
    "--output",
    "-o",
    default="calibration.xml",
    help="Output file for calibration parameters",
)
@click.option(
    "--min-stars", default=20, help="Minimum number of stars required per image"
)
@click.option("--star-threshold", default=100, help="Threshold for star detection")
def calibrate(
    image_paths: List[str],
    api_key: str,
    output: str,
    min_stars: int,
    star_threshold: float,
) -> None:
    """
    Generate fisheye camera calibration parameters from night sky images.

    IMAGE_PATHS: One or more paths to night sky images
    """
    calibrate_cli(image_paths, api_key, output, min_stars, star_threshold)


if __name__ == "__main__":
    main()
