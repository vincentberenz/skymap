import tempfile
from pathlib import Path, PosixPath
from subprocess import CalledProcessError, run
from typing import Optional

import imageio
import numpy as np
from astropy.wcs import WCS
from loguru import logger
from .image import ImageData
from .conversions import normalize_to_uint8, to_grayscale_8bits, stretch


def get_num_processes() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    cpu_count = 1 if cpu_count == 1 else cpu_count - 1
    return cpu_count


class AstrometryFailed(Exception):
    pass


class AstrometryError(Exception):
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int):
        self.stdout = stdout.decode("utf-8", errors="replace")
        self.stderr = stderr.decode("utf-8", errors="replace")
        self.returncode = returncode


class PlateSolving:

    @staticmethod
    def from_file(filepath: Path, cpulimit_seconds: int, working_dir: Path) -> WCS:
        """Run Astrometry.net's `solve-field` command and return the WCS

        Args:
            filepath: Path to the image file to be solved

        Returns:
            WCS object if solving was successful

        Raises:
            AstrometryFailed: If solving failed
            AstrometryError: If solving failed with an error
        """

        wcs_file = working_dir / f"{filepath.stem}.wcs"
        solved_file = working_dir / f"{filepath.stem}.solved"

        command = ["solve-field","--overwrite"]
        if cpulimit_seconds > 0:
            command.extend(["--cpulimit", str(cpulimit_seconds)])
        command.extend(["--dir", str(working_dir), "-o", str(filepath.stem), str(filepath)])

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
    def from_numpy(filename: str, image_data: ImageData, cpulimit_seconds: int, working_dir: Path) -> WCS:
        working_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_data.save(working_dir, filename)
        return PlateSolving.from_file(image_path, cpulimit_seconds, working_dir)
