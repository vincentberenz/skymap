import healpy as hp
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from PIL import Image
import healpy as hp
import matplotlib.pyplot as plt


# Load the image data from the TIFF file
image_file = 'patch_0111.tiff'
image = Image.open(image_file)
image_data = np.array(image)

# Load the WCS information from the .wcs file
wcs_file = 'patch_0111.wcs'
with fits.open(wcs_file) as hdul:
    wcs = WCS(hdul[0].header)

# Define the resolution of the HEALPix map
nside = 128  # Adjust based on your needs
npix = hp.nside2npix(nside)
healpix_map = np.zeros(npix)

# Iterate over each pixel in the image
for y in range(image_data.shape[0]):
    for x in range(image_data.shape[1]):
        # Get the pixel value
        pixel_value = image_data[y, x]

        # Convert pixel coordinates to celestial coordinates
        ra, dec = wcs.wcs_pix2world(x, y, 0)

        # Convert celestial coordinates to HEALPix index
        theta = np.radians(90 - dec)
        phi = np.radians(ra)
        hp_index = hp.ang2pix(nside, theta, phi)

        # Accumulate the pixel value into the HEALPix map
        healpix_map[hp_index] += pixel_value

# Save the HEALPix map to a file
hp.write_map('patch_0111_healpix.fits', healpix_map, overwrite=True)


# Load the HEALPix map from the FITS file
healpix_file = 'patch_0111_healpix.fits'
healpix_map = hp.read_map(healpix_file)

# Plot the HEALPix map
hp.mollview(healpix_map, title="HEALPix Map", unit="Intensity", cmap="viridis")
hp.graticule()

# Show the plot
plt.show()