import numpy as np
from typing import NamedTuple, Optional
from .patch_types import Size, Pixel
from ..image import ImageData

class PatchArgs(NamedTuple):
    """Container for patch creation arguments.

    Attributes:
        label: Label for the patch
        location: Tuple of (row, column) for the patch
        size: Tuple of (height, width) for the patch
        image_data: ImageData object containing the full image
        index: Unique identifier for the patch
        no_plate_solving: If True, skip plate solving
        cpulimit_seconds: CPU time limit for plate solving
    """

    label: str
    location: Pixel
    size: Size
    image_data: ImageData
    index: int
    no_plate_solving: bool
    cpulimit_seconds: Optional[int]
    working_dir: str

    def get_image(self)->ImageData:
        return self.image_data.get(self.location, self.size)