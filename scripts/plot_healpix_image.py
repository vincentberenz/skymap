from skymapper.patches import PatchedImage
from skymapper.healpix import load_healpix_dict, hammer_plot
from pathlib import Path

RECORD_PATH = Path("/home/vberenz/Workspaces/skymap-data/one-night/ns5/astrometry/patches/nightskycam5_2025_06_20_02_42_30.healpix.pkl.gz")
PLOT_PATH = Path("/home/vberenz/Workspaces/skymap-data/one-night/ns5/astrometry/patches/nightskycam5_2025_06_20_02_42_30.healpix.plot.png")

def main():

    record = load_healpix_dict(RECORD_PATH)
    hammer_plot(record, PLOT_PATH)
    #hammer_plot(record, None)

if __name__ == "__main__":
    main()
