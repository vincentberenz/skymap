"""
Patches module for handling image patches and their WCS solutions.
"""

from .patch import Patch
from .patch_args import PatchArgs
from .patch_coords import PatchCoords
from .patched_image import PatchedImage

__all__ = [
    'Patch',
    'PatchArgs',
    'PatchCoords',
    'PatchedImage',
]