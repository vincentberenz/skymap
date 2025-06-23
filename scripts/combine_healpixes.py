import numpy as np
from skymapper.patches import PatchedImage, HealpixPatchedImage
import healpy as hp
import matplotlib.pyplot as plt

INPUT_PATCHES = "images/nightskycam3_2025_04_05_03_54_30.pkl.gz"
OUTPUT_NUMPY = "images/nightskycam3_2025_04_05_03_54_30.npy"
OUTPUT_HEALPIX = "images/nightskycam3_2025_04_05_03_54_30.fits"
NSIDE = 512*32
PATCHES: Optional[tuple[int,...]] = (77,78,63)

if __name__ == "__main__":
    
    patches = PatchedImage.load(INPUT_PATCHES)
    healpix_map: np.ndarray[tuple[int], np.float32] = patches.to_healpix(NSIDE, PATCHES)
    # save as numpy array
    np.save(OUTPUT_NUMPY, healpix_map)
    # save as healpix file
    hp.write_map(OUTPUT_HEALPIX, healpix_map, overwrite=True)


    # Load the HEALPix map from the FITS file
    healpix_map = hp.read_map(OUTPUT_HEALPIX)

    # Plot the HEALPix map
    hp.mollview(healpix_map, title="HEALPix Map", unit="Intensity", cmap="viridis")
    hp.graticule()

    # Show the plot
    plt.show()