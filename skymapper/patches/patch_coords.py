from dataclasses import dataclass
import numpy as np
from loguru import logger

from .patch_types import Pixel, Size, Shape


@dataclass
class PatchCoords:
    location: Pixel
    size: Size

    @classmethod
    def create(cls, shape: Shape, patch_size: Size)->list["PatchCoords"]:

        patch_height, patch_width = patch_size
        height, width = shape[:2]

        # Calculate number of patches needed in each dimension
        num_patches_y = max(1, (height + patch_height - 1) // patch_height)
        num_patches_x = max(1, (width + patch_width - 1) // patch_width)
        
        # Calculate required overlap to ensure full coverage
        if num_patches_y > 1:
            overlap_y = (patch_height * num_patches_y - height) / (num_patches_y - 1)
            overlap_y = int(np.ceil(overlap_y))  # Round up to ensure coverage
            step_y = patch_height - overlap_y
        else:
            step_y = 0
            
        if num_patches_x > 1:
            overlap_x = (patch_width * num_patches_x - width) / (num_patches_x - 1)
            overlap_x = int(np.ceil(overlap_x))  # Round up to ensure coverage
            step_x = patch_width - overlap_x
        else:
            step_x = 0
            
        logger.debug(f"Using patch size: {patch_size}")
        logger.debug(f"Image size: {shape}")
        logger.debug(f"Number of patches: {num_patches_y}x{num_patches_x}")
        logger.debug(f"Computed overlap: y={overlap_y}, x={overlap_x}")
        logger.debug(f"Step sizes: y={step_y}, x={step_x}")

        # Generate all possible patch coordinates
        patches = []
        index = 0
        for y in range(0, height - patch_height + 1, step_y):
            for x in range(0, width - patch_width + 1, step_x):
                # Calculate actual patch dimensions (may be smaller at edges)
                actual_height = min(patch_height, height - y)
                actual_width = min(patch_width, width - x)
                
                patches.append(cls((y,x), (actual_height, actual_width)))
                index += 1
                
                # If we've reached the end of the row, break to avoid extra patches
                if x + step_x >= width - patch_width + 1:
                    break
                
            # If we've reached the end of the column, break to avoid extra patches
            if y + step_y >= height - patch_height + 1:
                break
        
        return patches