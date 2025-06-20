import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from astropy.wcs import WCS
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D

mplstyle.use("fast")


def create_spherical_projection(image_data: np.ndarray, wcs: WCS):
    """
    Project an image onto a 3D sphere using its WCS information.

    Parameters:
    -----------
    image_data : numpy.ndarray
        The 2D image array
    wcs : astropy.wcs.WCS
        WCS object containing the coordinate transformation

    Returns:
    --------
    tuple of (x, y, z, values) for 3D plotting
    """
    # Create coordinate grid
    y, x = np.indices(image_data.shape)

    # Convert to world coordinates (RA/Dec)
    ra, dec = wcs.all_pix2world(x, y, 0)

    # Convert to 3D Cartesian coordinates
    dec_rad = np.deg2rad(dec)
    ra_rad = np.deg2rad(ra)

    # Create 3D coordinates on unit sphere
    x = np.cos(ra_rad) * np.cos(dec_rad)
    y = np.sin(ra_rad) * np.cos(dec_rad)
    z = np.sin(dec_rad)

    return x, y, z, image_data


def plot_spherical_projection(x, y, z, values):
    """
    Create a 3D visualization of the projected data.

    Parameters:
    -----------
    x, y, z : numpy.ndarray
        3D coordinates of points
    values : numpy.ndarray
        Image values at each point
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface with a colormap
    surf = ax.scatter(
        x.flatten(),
        y.flatten(),
        z.flatten(),
        c=values.flatten(),
        cmap="viridis",
        alpha=0.6,
    )

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Image Projected onto Celestial Sphere")

    # Add a colorbar
    plt.colorbar(surf, ax=ax, label="Intensity")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()


def main():
    image_path = "/tmp/astrometry/processed_patch.tiff"
    wcs = WCS("/tmp/astrometry/processed_patch.wcs")
    image_data = imread(image_path)
    x, y, z, values = create_spherical_projection(image_data, wcs)
    plot_spherical_projection(x, y, z, values)


if __name__ == "__main__":
    main()
