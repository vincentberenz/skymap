from scripts.process_image import NUM_PROCESSES
from skymapper.patches import PatchedImage
from skymapper.healpix import overlay_healpix_indices, HEALPixNside
from skymapper.conversions import stretch
from pathlib import Path    
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import imageio
import multiprocessing as mp

BASE_FOLDER = Path("/home/vberenz/Workspaces/skymap-data/june_19th_2025/ns3/")
PATCHED_IMAGES = BASE_FOLDER / "astrometry/"
#PATCHED_IMAGES = BASE_FOLDER / "test/"
PATCHED_IMAGES_EXTENSION = ".pkl.gz"
HEALPIX_PATH = BASE_FOLDER / "healpix/"
NSIDE = HEALPixNside(8)
NUM_PROCESSES = 7

def one_file(nside: HEALPixNside, path: str, output_path: str):
    logger.info(f"Computing healpix indices for {path.stem}")
    patched_image = PatchedImage.load(path)
    indices: np.ndarray = patched_image.get_healpix_indices(nside)
    np.save(output_path / f"{path.stem}.indices.nside{nside}.npy", indices)   
    overlayed_rgb = overlay_healpix_indices(NSIDE, stretch(patched_image.image), indices)        
    imageio.imwrite(output_path / f"{path.stem}.overlayed.tiff", overlayed_rgb)

def all_files(nside: HEALPixNside, path: Path,  patched_images_extension: str, output_path: Path, num_processes: int):
    files = list(path.glob("*" + patched_images_extension))
    if num_processes > 1:
        with mp.Pool(processes=num_processes) as pool:
            pool.starmap(one_file, [(nside, f, output_path) for f in files])
    else:
        for f in files:
            one_file(nside, f, output_path)

if __name__ == "__main__":
    all_files(NSIDE, PATCHED_IMAGES, PATCHED_IMAGES_EXTENSION, HEALPIX_PATH, NUM_PROCESSES)
