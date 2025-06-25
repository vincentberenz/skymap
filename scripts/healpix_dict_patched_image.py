from skymapper.patches import PatchedImage
from skymapper.healpix import healpix_record_info
from pathlib import Path

PATCHED_IMAGE = "/home/vberenz/Workspaces/skymap-data/one-night/ns3/astrometry/patches/nightskycam3_2025_06_20_02_42_30.pkl.gz"
OUTPUT_FILE = "/home/vberenz/Workspaces/skymap-data/one-night/ns3/astrometry/patches/nightskycam3_2025_06_20_02_42_30.healpix.pkl.gz"
ONE_PATCH_ONLY = False
NSIDE = 1024
NUM_PROCESSES = 7

def main():

    patched_image = PatchedImage.load(PATCHED_IMAGE)

    if ONE_PATCH_ONLY:
        patch_indices = [patched_image.get_patch_indices()[0]]        
    else:
        patch_indices = None

    record = patched_image.get_healpix_record(NSIDE, num_processes=NUM_PROCESSES, output_path=Path(OUTPUT_FILE), patch_indices=patch_indices)
    healpix_record_info(record)

if __name__ == "__main__":
    main()
    