import numpy as np
from typing import NamedTuple, Optional
from .patch_types import Size, Pixel

class PatchArgs(NamedTuple):
    """Container for patch creation arguments.

    Attributes:
        i: Starting row index of the patch
        j: Starting column index of the patch
        image: The full image array
        patch_size: Tuple of (height, width) for the patch
        index: Unique identifier for the patch
        debug_folder: Optional folder for debug output
        no_plate_solving: If True, skip plate solving
        cpulimit_seconds: CPU time limit for plate solving
    """

    location: Pixel
    size: Size
    image: np.ndarray
    index: int
    no_plate_solving: bool
    cpulimit_seconds: Optional[int]
    working_dir: str
