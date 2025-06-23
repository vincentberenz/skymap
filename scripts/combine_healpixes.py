import numpy as np
from skymapper.patches import PatchedImage
import healpy as hp
from typing import Optional
import matplotlib.pyplot as plt
from loguru import logger

INPUT_PATCHES = "images/nightskycam3_2025_04_05_03_54_30.pkl.gz"
OUTPUT_NUMPY = "images/nightskycam3_2025_04_05_03_54_30.npy"
OUTPUT_HEALPIX = "images/nightskycam3_2025_04_05_03_54_30.fits"
NSIDE = 512*32
#PATCHES: Optional[tuple[int,...]] = (77,78,63)
PATCHES: Optional[tuple[int,...]] = (77,)

if __name__ == "__main__":
    
    patches = PatchedImage.load(INPUT_PATCHES)

    logger.info("Creating the healpix map")
    healpix_map: np.ndarray[tuple[int], np.uint8] = patches.to_healpix(NSIDE, PATCHES)
    logger.info("Saving the healpix map (numpy)")
    # save as numpy array
    np.save(OUTPUT_NUMPY, healpix_map)
    logger.info("Saving the healpix map (healpix)")
    # save as healpix file
    hp.write_map(OUTPUT_HEALPIX, healpix_map, overwrite=True)


    # Load the HEALPix map from the FITS file
    logger.info("Loading the healpix map")
    healpix_map = hp.read_map(OUTPUT_HEALPIX)

    # Plot the HEALPix map
    logger.info("Plotting the healpix map")
    hp.mollview(healpix_map, title="HEALPix Map", unit="Intensity", cmap="viridis")
    hp.graticule()

    # Show the plot
    logger.info("Showing the plot")
    plt.show()