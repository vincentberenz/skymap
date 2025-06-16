import sys
from pathlib import Path
from typing import List, Optional

import click
import numpy as np

from .calibration import generate_calibration_from_stars
from .logger import logger
from .mapper import process_image
from .patches import display_patches, extract_patches_from_file
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
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("calibration_file", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
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
    CALIBRATION_FILE: Path to the calibration file
    OUTPUT: Output directory for results
    """
    logger.info(f"Processing image {image_path}...")
    try:
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
@click.argument(
    "patch_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--delay",
    type=int,
    default=0,
    show_default=True,
    help="Delay in milliseconds between patches (0 = wait for key press)",
)
def show_patches(patch_dir: str, delay: int) -> None:
    """
    Display all patches in a directory one by one with their metadata.

    PATCH_DIR: Directory containing patch files (.npz)
    """
    try:
        display_patches(patch_dir, delay)
    except Exception as e:
        logger.error(f"Error displaying patches: {str(e)}")
        raise click.ClickException(str(e))


@main.command()
@click.argument("image_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o",
    "--output",
    default="patches",
    show_default=True,
    type=click.Path(dir_okay=True),
)
@click.option(
    "-s",
    "--patch-size",
    default=500,
    show_default=True,
    help="Size of image patches",
)
@click.option(
    "-p",
    "--patch-overlap",
    default=250,
    show_default=True,
    help="Overlap between patches",
)
@click.option(
    "-m",
    "--min-stars",
    default=5,
    show_default=True,
    help="Minimum number of stars per patch",
)
@click.option(
    "-t",
    "--star-threshold",
    default=20,
    show_default=True,
    help="Star detection threshold",
)
def generate_patches(
    image_dir: str,
    output: str,
    patch_size: int,
    patch_overlap: int,
    min_stars: int,
    star_threshold: int,
) -> None:
    """
    Generate and save image patches for calibration.

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

        logger.info(f"Generating patches from {len(image_paths)} images")
        total_patches = 0
        for img_path in image_paths:
            total_patches += extract_patches_from_file(
                img_path,
                patch_size,
                patch_overlap,
                output,
                min_stars=min_stars,
                star_threshold=star_threshold,
            )
        logger.info(f"Generated a total number of {total_patches} patches")
    except click.UsageError as e:
        logger.error(f"Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating patches: {str(e)}")
        raise


@main.command()
@click.argument("patch_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-k",
    "--api-key",
    help="Astrometry.net API key (required for plate solving)",
    required=True,
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
def process_patches(
    patch_dir: str,
    api_key: str,
    output: str,
    min_stars: int,
    star_threshold: float,
) -> None:
    """
    Process saved patches to generate fisheye camera calibration parameters.

    PATCH_DIR: Directory containing saved patches
    """
    try:
        if not api_key:
            raise click.UsageError("API key is required for plate solving")

        logger.info(f"Processing patches from {patch_dir}")
        process_patches(
            patch_dir=patch_dir,
            api_key=api_key,
            output_file=output,
            min_stars=min_stars,
            star_threshold=star_threshold,
        )
    except click.UsageError as e:
        logger.error(f"Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing patches: {str(e)}")
        raise


@main.command()
@click.argument("image_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-k",
    "--api-key",
    help="Astrometry.net API key (required for plate solving)",
    required=True,
)
@click.option(
    "-o",
    "--output",
    default="skymapper_fisheye_calibration.xml",
    show_default=True,
    type=click.Path(dir_okay=False),
)
@click.option(
    "-p",
    "--patch-size",
    default=1024,
    show_default=True,
    help="Size of image patches",
)
@click.option(
    "-l",
    "--patch-overlap",
    default=256,
    show_default=True,
    help="Overlap between patches",
)
def calibrate(
    image_dir: str,
    api_key: str,
    output: str,
    min_stars: int,
    star_threshold: float,
    patch_size: int,
    patch_overlap: int,
) -> None:
    """
    Generate fisheye camera calibration parameters from night sky images.

    IMAGE_DIR: Path to directory containing TIFF images
    """
    try:
        # Find all TIFF images in the directory (both .tiff and .tif extensions)
        tiff_files = list(Path(image_dir).glob("*.tiff")) + list(
            Path(image_dir).glob("*.tif")
        )

        if not tiff_files:
            raise click.UsageError(f"No TIFF images found in directory: {image_dir}")

        logger.info(f"Found {len(tiff_files)} TIFF images in {image_dir}")

        # Convert to string list
        image_paths = [str(f) for f in tiff_files]

        logger.info(
            f"Starting calibration with {len(image_paths)} images, "
            f"min_stars={min_stars}, patch_size={patch_size}, "
            f"patch_overlap={patch_overlap}"
        )

        # Generate calibration using the plate solver's star detection
        generate_calibration_from_stars(
            image_paths=image_paths,
            api_key=api_key,
            output_file=output,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )

        logger.info(f"Successfully generated calibration file: {output}")

    except click.UsageError as e:
        logger.error(f"Usage error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating calibration: {str(e)}")
        raise


if __name__ == "__main__":
    main()
