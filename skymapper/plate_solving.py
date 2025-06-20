import tempfile
from pathlib import Path
from subprocess import CalledProcessError, run

import imageio
import numpy as np
from astropy.wcs import WCS
from loguru import logger
from multipledispatch import dispatch

from .conversions import to_grayscale_8bits


class AstrometryFailed(Exception):
    pass


class AstrometryError(Exception):
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int):
        self.stdout = stdout.decode("utf-8", errors="replace")
        self.stderr = stderr.decode("utf-8", errors="replace")
        self.returncode = returncode


@dispatch(Path)
def plate_solving(filepath: Path) -> WCS:
    """Run Astrometry.net's `solve-field` command and return the WCS

    Args:
        filepath: Path to the image file to be solved

    Returns:
        WCS object if solving was successful

    Raises:
        AstrometryFailed: If solving failed
        AstrometryError: If solving failed with an error
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        solved_file = Path(tmp_dir) / "astrometry.solved"
        wcs_file = Path(tmp_dir) / "astrometry.wcs"

        try:
            run(
                ["solve-field", "--dir", tmp_dir, "-o", "astrometry", str(filepath)],
                capture_output=True,
                check=True,
            )
            if solved_file.exists():
                return WCS(wcs_file)
            else:
                raise AstrometryFailed()
        except CalledProcessError as e:
            raise AstrometryError(e.stdout, e.stderr, e.returncode)


@dispatch(np.ndarray)
def plate_solving(image: np.ndarray) -> WCS:
    img = to_grayscale_8bits(image)

    # saving image to temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "image.tiff"
        imageio.imwrite(image_path, img, format="tiff")
        return plate_solving(image_path)
