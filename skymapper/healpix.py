from typing import NewType
import numpy as np
from loguru import logger
from pathlib import Path
import pickle
import gzip
import healpy as hp

HEALPixIndex = NewType("HEALPixIndex", int)
HEALPixNside = NewType("HEALPixNside", int)
HEALPixDict = NewType("HEALPixDict", dict[HEALPixIndex, np.ndarray])
HEALPixDictRecord = NewType("HEALPixDictRecord", tuple[HEALPixNside, HEALPixDict])

def save_healpix_dict(nside: HEALPixNside, healpix_dict: HEALPixDict, output_path: Path)->None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving HEALPix dictionary to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump((nside, healpix_dict), f)
    logger.info(f"Saved HEALPix dictionary to: {output_path}")


def load_healpix_dict(path: Path)->HEALPixDictRecord:
    logger.info(f"Loading HEALPix dictionary from: {path}")
    with open(path, "rb") as f:
        nside, healpix_dict = pickle.load(f)
    logger.info(f"Loaded HEALPix dictionary from: {path}")
    return HEALPixDictRecord((nside, healpix_dict))

def num_pixels(nside: HEALPixNside)->int:
    return hp.nside2npix(nside)

def healpix_record_info(record: HEALPixDictRecord)->None:
    nside, healpix_dict = record
    npixels = num_pixels(nside)
    logger.info(f"HEALPix dictionary with {len(healpix_dict)} / {npixels} pixels and nside {nside}")