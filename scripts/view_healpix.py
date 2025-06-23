import healpy as hp
import matplotlib.pyplot as plt
from loguru import logger

if __name__ == "__main__":
    
    #HEALPIX_FILE = "images/nightskycam3_2025_04_05_03_54_30.fits"3
    HEALPIX_FILE = "images/earth-8192.fits"

    # read the healpix file and display it with a cartview
    logger.info(f"Loading healpix file: {HEALPIX_FILE}")
    healpix_map = hp.read_map(HEALPIX_FILE)
    logger.info(f"Computing cartesian view")
    hp.cartview(healpix_map, title="HEALPix Map", unit="Intensity", cmap="viridis")
    logger.info(f"Adding graticule")
    hp.graticule()
    logger.info(f"Showing plot")
    plt.show()