import sys
from pathlib import Path
from typing import List, Optional

import click
import numpy as np

from .calibration import generate_calibration_from_stars
from .logger import logger
from .mapper import (
    convert_to_healpix,
    correct_fisheye_distortion,
    load_image,
    process_image,
)
from .plate_solver import solve_plate
from .utils import plot_healpix_projection, visualize_healpix_map


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
        show_plot = not no_show
        healpix_map = process_tiff_image(
            image_path=image_path,
            healpix_nside=nside,
            calibration_file=calibration,
            api_key=api_key,
            output_path=output,
            show_plot=show_plot,
        )
        if show_plot:
            plot_healpix_projection(healpix_map)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


@main.command()
@click.argument("image-path", type=click.Path(exists=True))
@click.option(
    "-c", "--calibration", required=True, help="Path to fisheye calibration parameters"
)
@click.option("-o", "--output", required=True, help="Output file path")
@click.option("-a", "--api-key", help="Astrometry.net API key")
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
    try:
        logger.info(f"Processing image: {image_path}")
        healpix_map = process_tiff_image(
            image_path=image_path,
            calibration_file=calibration_file,
            api_key=api_key,
            output_path=output,
            show_plot=False,
        )
        logger.info(f"Successfully processed image and saved to {output}")
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


@main.command()
@click.argument("image_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-k", "--api-key", help="Astrometry.net API key (required for plate solving)"
)
@click.option(
    "-o",
    "--output",
    default="skymapper_fisheye_calibration.xml",
    show_default=True,
    type=click.Path(dir_okay=False),
)
@click.option(
    "-s",
    "--min-stars",
    default=20,
    show_default=True,
    help="Minimum number of stars required per image patch",
)
@click.option(
    "-t",
    "--star-threshold",
    default=100.0,
    show_default=True,
    help="Threshold for star detection (higher value = fewer stars)",
)
def calibrate(
    image_dir: str,
    api_key: str,
    output: str,
    min_stars: int,
    star_threshold: float,
) -> None:
    """
    Generate fisheye camera calibration parameters from night sky images.

    IMAGE_DIR: Path to directory containing TIFF images
    """
    try:
        # Find all TIFF images in the directory
        tiff_files = list(Path(image_dir).glob("*.tiff")) + list(
            Path(image_dir).glob("*.tif")
        )

        if not tiff_files:
            raise click.UsageError(f"No TIFF images found in directory: {image_dir}")

        logger.info(f"Found {len(tiff_files)} TIFF images in {image_dir}")

        # Convert to string list
        image_paths = [str(f) for f in tiff_files]

        # Rest of the function remains the same
        if not api_key:
            raise click.UsageError("API key is required for plate solving")

        logger.info(f"Generating calibration from {len(image_paths)} images")
        generate_calibration_from_stars(
            image_paths=image_paths,
            api_key=api_key,
            output_file=output,
            min_stars=min_stars,
            star_threshold=star_threshold,
        )
    except click.UsageError as e:
        logger.error(f"Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating calibration: {str(e)}")
        raise
        if not api_key:
            raise click.UsageError("API key is required for plate solving")

        logger.info(f"Generating calibration from {len(image_paths)} images")
        generate_calibration_from_stars(
            image_paths=image_paths,
            api_key=api_key,
            output_file=output,
            min_stars=min_stars,
            star_threshold=star_threshold,
        )


if __name__ == "__main__":
    main()
