import healpy as hp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    HEALPIX_FILE = "images/nightskycam3_2025_04_05_03_54_30.fits"

    # read the healpix file and display it with a cartview
    healpix_map = hp.read_map(HEALPIX_FILE)
    hp.cartview(healpix_map, title="HEALPix Map", unit="Intensity", cmap="viridis")
    hp.graticule()
    plt.show()