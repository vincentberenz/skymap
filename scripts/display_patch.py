import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle
from skymapper.patches import PatchedImage, combine_images
from loguru import logger
from pathlib import Path
from skymapper.conversions import to_grayscale_8bits, stretch

# Configuration - Update this path to point to your saved PatchedImage file
INPUT_FILE = Path("images/nightskycam3_2025_04_05_03_54_30.pkl.gz")
PATCH_INDICES = (63,77,78)




def plot_spherical_representation(data: np.ndarray, wcs: WCS):
    """
    Plots a spherical representation of the given data using the provided WCS.

    Parameters:
    - data: np.ndarray, the 2D array representing the image data.
    - wcs: WCS, the World Coordinate System object for the data.
    """
    # Determine the center of the image in pixel coordinates
    center_pixel = np.array([data.shape[0] // 2, data.shape[1] // 2])

    # Convert the center pixel to celestial coordinates
    center_sky = wcs.pixel_to_world(center_pixel[0], center_pixel[1])

    # Determine the corners of the image in pixel coordinates
    corner_pixels = np.array([[0, 0], [0, data.shape[1] - 1], [data.shape[0] - 1, 0], [data.shape[0] - 1, data.shape[1] - 1]])

    # Convert the corner pixels to celestial coordinates
    corner_sky = wcs.pixel_to_world(corner_pixels[:, 0], corner_pixels[:, 1])

    # Calculate the radius as the maximum angular distance from the center to the corners
    radius = max(center_sky.separation(corner_sky))

    # Create a figure with a WCS projection
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=wcs)

    # Plot the data
    ax.imshow(data, origin='lower', cmap='gray')

    # Set the axis labels
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    # Add a grid
    ax.grid(color='white', ls='solid')

    # Add a spherical circle to represent the data on the sphere
    circle = SphericalCircle((center_sky.ra, center_sky.dec), radius, edgecolor='red', facecolor='none', transform=ax.get_transform('icrs'))
    ax.add_patch(circle)

    plt.show()


if __name__ == "__main__":

    logger.info(f"Loading PatchedImage from: {INPUT_FILE}")

    patched_image = PatchedImage.load(INPUT_FILE)
    full_image = patched_image.image
    full_stretched_image = stretch(to_grayscale_8bits(full_image))

    # plotting one patch
    patch = patched_image.get_patch(PATCH_INDICES[0])
    patch_image = patch.get_image(full_stretched_image)
    plot_spherical_representation(patch_image, patch.wcs)

    # plotting several patches
    patches = [patched_image.get_patch(i) for i in PATCH_INDICES]
    combined_image, combined_wcs = combine_images(patches, full_image)
    plot_spherical_representation(combined_image, combined_wcs)
    