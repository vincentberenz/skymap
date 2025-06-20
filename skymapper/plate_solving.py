import tempfile
from pathlib import Path, PosixPath
from subprocess import CalledProcessError, run
from typing import Optional

import imageio
import numpy as np
from astropy.wcs import WCS
from loguru import logger

from .conversions import to_grayscale_8bits


class AstrometryFailed(Exception):
    pass


class AstrometryError(Exception):
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int):
        self.stdout = stdout.decode("utf-8", errors="replace")
        self.stderr = stderr.decode("utf-8", errors="replace")
        self.returncode = returncode


class PlateSolving:

    @staticmethod
    def from_file(filepath: Path, cpulimit_seconds: int) -> WCS:
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

            command = ["solve-field"]
            if cpulimit_seconds > 0:
                command.extend(["--cpulimit", str(cpulimit_seconds)])
            command.extend(["--dir", tmp_dir, "-o", "astrometry", str(filepath)])

            try:
                run(
                    command,
                    capture_output=True,
                    check=True,
                )
                if solved_file.exists():
                    return WCS(str(wcs_file))
                else:
                    raise AstrometryFailed()
            except CalledProcessError as e:
                raise AstrometryError(e.stdout, e.stderr, e.returncode)

    @staticmethod
    def from_numpy(image: np.ndarray, cpulimit_seconds: int) -> WCS:
        img = to_grayscale_8bits(image)

        # saving image to temporary file
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "image.tiff"
            imageio.imwrite(image_path, img, format="tiff")
            return PlateSolving.from_file(image_path, cpulimit_seconds)
