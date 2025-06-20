import gzip
import multiprocessing as mp
import os
import pickle
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NamedTuple, NewType, Optional, Tuple

import cv2
import imageio
import numpy as np
from astropy.wcs import WCS
from loguru import logger
from multipledispatch import dispatch


from .conversions import to_grayscale_8bits, stretch
from .plate_solving import AstrometryError, AstrometryFailed, PlateSolving

# Define a named tuple for patch arguments
class PatchArgs(NamedTuple):
    """Container for patch creation arguments.

    Attributes:
        i: Starting row index of the patch
        j: Starting column index of the patch
        image: The full image array
        patch_size: Size of the patch (width and height)
        index: Unique identifier for the patch
    """

    i: int
    j: int
    image: np.ndarray
    patch_size: int
    index: int
    debug_folder: Optional[str]
    no_plate_solving: bool
    cpulimit_seconds: Optional[int]


Pixel = NewType("Pixel", tuple[int, int])


def get_num_processes() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    cpu_count = 1 if cpu_count == 1 else cpu_count - 1
    return cpu_count


@dataclass
class Patch:
    index: int
    location: Pixel
    size: tuple[int, int]
    wcs: Optional[WCS]

    def inside(self, pixel: Pixel) -> bool:
        return (
            self.location[0] <= pixel[0] < self.location[0] + self.size[0]
            and self.location[1] <= pixel[1] < self.location[1] + self.size[1]
        )

    def distance_to_center(self, pixel: Pixel) -> float:
        return np.sqrt(
            (pixel[0] - self.location[0]) ** 2 + (pixel[1] - self.location[1]) ** 2
        )

    def get_center(self) -> Pixel:
        return Pixel(
            (self.location[0] + self.size[0] // 2, self.location[1] + self.size[1] // 2)
        )


def _create_patch(patch_args: PatchArgs) -> Patch:
    """Helper function to create and solve a single patch.

    This is a module-level function to support multiprocessing.

    Args:
        patch_args: Named tuple containing patch creation arguments

    Returns:
        Patch object with WCS solution if successful, None otherwise
    """

    patch_data = patch_args.image[
        patch_args.i : patch_args.i + patch_args.patch_size,
        patch_args.j : patch_args.j + patch_args.patch_size,
    ]

    if patch_args.debug_folder is not None:
        p = Path(patch_args.debug_folder)
        p.mkdir(parents=True, exist_ok=True)
        patch_path = p / f"patch_{patch_args.index}_{patch_args.i}_{patch_args.j}.tiff"
        logger.debug(f"saving patch {patch_args.index} {patch_args.i} {patch_args.j} to {p}")
        imageio.imwrite(
            patch_path,
            patch_data,
        )

    logger.info(
        f"Processing patch {patch_args.index} / {patch_data.shape} / {patch_data.dtype}"
    )

    if patch_args.no_plate_solving:
        logger.info(f"Plate solving disabled, skipping")
        return Patch(
            location=(patch_args.i, patch_args.j),
            size=(patch_args.patch_size, patch_args.patch_size),
            index=patch_args.index,
            wcs=None,
        )

    try:
        # Solve the plate for this patch
        cpulimit = (
            0 if patch_args.cpulimit_seconds is None else patch_args.cpulimit_seconds
        )

        wcs = PlateSolving.from_numpy(patch_data, cpulimit)

        logger.info(f"Successfully solved patch {patch_args.index}")
        return Patch(
            location=Pixel((patch_args.i, patch_args.j)),
            size=(patch_args.patch_size, patch_args.patch_size),
            index=patch_args.index,
            wcs=wcs,
        )
    except (AstrometryError, AstrometryFailed) as e:
        logger.warning(
            f"Failed to solve patch {patch_args.index} at "
            f"({patch_args.i}, {patch_args.j}): {str(e)}"
        )
        # Return a patch without WCS if solving fails
        return Patch(
            location=(patch_args.i, patch_args.j),
            size=(patch_args.patch_size, patch_args.patch_size),
            index=patch_args.index,
            wcs=None,
        )


@dataclass
class PatchedImage:
    patches: list[Patch]
    image: np.ndarray

    def display(self, path: Path, border_thickness: int = 2, text_scale: float = 0.8, text_thickness: int = 2, apply_stretch: bool = True) -> None:
        """
        Dump into path a tiff image encoding the image attribute on which
        the border of the patches are displayed. The borders are displayed in green
        for solved patches and red for unsolved patches, with patch indices in the center.

        Args:
            path: Path where to save the visualization image
            border_thickness: Thickness of the border in pixels
            text_scale: Font scale factor for the index numbers
            text_thickness: Thickness of the text for index numbers

        Raises:
            IOError: If there's an error writing the image file
        """

        if apply_stretch:
            vis_image = stretch(self.image)
        else:
            vis_image = self.image

        vis_image = to_grayscale_8bits(vis_image)

        
        # Convert to BGR color space for OpenCV
        if len(vis_image.shape) == 2:  # Grayscale
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        elif vis_image.shape[2] == 1:  # Single channel
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        elif vis_image.shape[2] == 4:  # RGBA
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2BGR)
        elif vis_image.shape[2] == 3:  # RGB
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    
        # Draw borders and indices for each patch
        for patch in self.patches:
            i, j = patch.location
            h, w = patch.size
            
            # Determine colors (green for solved, red for unsolved)
            # OpenCV uses BGR color order
            is_solved = patch.wcs is not None
            color = (0, 255, 0) if is_solved else (0, 0, 255)  # Green for solved, red for unsolved
            text_color = color  # Use the same color as the border
            
            logger.debug(f"Drawing border for patch {patch.index} at ({i}, {j}) with color {color}")
            
            # Draw rectangle borders
            # Top border
            cv2.rectangle(vis_image, 
                        (j, i),  # Top-left corner
                        (j + w - 1, i + border_thickness - 1),  # Bottom-right corner
                        color, 
                        -1)  # Filled rectangle
            
            # Bottom border
            cv2.rectangle(vis_image, 
                        (j, i + h - border_thickness), 
                        (j + w - 1, i + h - 1), 
                        color, 
                        -1)
            
            # Left border
            cv2.rectangle(vis_image, 
                        (j, i + border_thickness), 
                        (j + border_thickness - 1, i + h - border_thickness - 1), 
                        color, 
                        -1)
            
            # Right border
            cv2.rectangle(vis_image, 
                        (j + w - border_thickness, i + border_thickness), 
                        (j + w - 1, i + h - border_thickness - 1), 
                        color, 
                        -1)
            
            # Add index number in the center of the patch
            text = str(patch.index)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)[0]
            text_x = j + (w - text_size[0]) // 2
            text_y = i + (h + text_size[1]) // 2
            
            # Add a small padding around the text for better visibility
            padding = 5
            cv2.rectangle(vis_image,
                        (text_x - padding, text_y - text_size[1] - padding),
                        (text_x + text_size[0] + padding, text_y + padding),
                        (0, 0, 0),  # Black background
                        -1)
            
            # Draw the index number
            cv2.putText(vis_image, 
                       text, 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       text_scale, 
                       text_color, 
                       text_thickness)

        # Save the visualization
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), vis_image)

    def dump(self, path: Path) -> None:
        """
        Serialize the PatchedImage instance to a file using pickle.

        Args:
            path: Path where to save the serialized PatchedImage

        Raises:
            IOError: If there's an error writing to the file
            pickle.PickleError: If serialization fails
        """

        try:
            # Create parent directories if they don't exist
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Use gzip compression for smaller file size
            with gzip.open(path, "wb") as f:
                pickle.dump(
                    {
                        "patches": self.patches,
                        "image": self.image,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            logger.info(f"Successfully saved PatchedImage to {path} ({len(self.patches)} patches)")

        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to save PatchedImage to {path}: {str(e)}")
            raise

    @classmethod
    def load(cls, path: Path) -> "PatchedImage":
        """
        Deserialize a PatchedImage instance from a file.

        Args:
            path: Path to the serialized PatchedImage file

        Returns:
            PatchedImage: The deserialized PatchedImage instance

        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
            pickle.PickleError: If deserialization fails
            KeyError: If the file is missing required data
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            with gzip.open(path, "rb") as f:
                data = pickle.load(f)

            # Validate the loaded data
            if not all(key in data for key in ["patches", "image"]):
                raise KeyError("Loaded data is missing required fields")

            logger.info(f"Successfully loaded PatchedImage from {path} ({len(data['patches'])} patches)")
            return cls(patches=data["patches"], image=data["image"])

        except (IOError, pickle.PickleError, KeyError) as e:
            logger.error(f"Failed to load PatchedImage from {path}: {str(e)}")
            raise

    def get_patch(self, pixels: Pixel) -> Patch:
        candidates = [patch for patch in self.patches if patch.inside(pixels)]
        if not candidates:
            raise ValueError(f"No patch found for pixels {pixels}")
        return min(candidates, key=lambda patch: patch.distance_to_center(pixels))

    @dispatch(int)
    def get_image(self, patch_index: int) -> np.ndarray:
        patch = self.patches[patch_index]
        return self.image[
            patch.location[0] : patch.location[0] + patch.size[0],
            patch.location[1] : patch.location[1] + patch.size[1],
        ]

    @dispatch(tuple)  # Changed from Pixel to tuple to avoid NewType dispatch issues
    def get_image(self, pixel: tuple[int, int]) -> np.ndarray:
        patch = self.get_patch(pixel)
        return self.get_image(patch.index)

    @classmethod
    def from_file(
        cls,
        path: Path,
        patch_size: int,
        patch_overlap: int,
        num_processes: Optional[int] = get_num_processes(),
        debug_folder: Optional[Path] = None,
        no_plate_solving: bool = False,
        cpulimit_seconds: Optional[int] = None,
    ) -> "PatchedImage":
        # read an image file and calls from_image
        image = imageio.imread(path)
        return cls.from_image(
            image,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            num_processes=num_processes,
            debug_folder=debug_folder,
            no_plate_solving=no_plate_solving,
            cpulimit_seconds=cpulimit_seconds,
        )

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        patch_size: int,
        patch_overlap: int,
        num_processes: Optional[int] = get_num_processes(),
        debug_folder: Optional[Path] = None,
        no_plate_solving: bool = False,
        cpulimit_seconds: Optional[int] = None,
    ) -> "PatchedImage":
        """
        Create a PatchedImage by dividing the input image into overlapping patches.
        Uses multiprocessing to speed up patch creation and plate solving.

        Args:
            image: Input image as a numpy array
            patch_size: Size of each square patch in pixels
            patch_overlap: Overlap between adjacent patches in pixels
            num_processes: Number of processes to use for parallel processing.
                        If None, uses all available CPU cores.

        Returns:
            PatchedImage instance containing all patches with WCS solutions

        Raises:
            AstrometryError: If plate solving fails for any patch
        """
        if num_processes is None:
            num_processes = 1

        img = to_grayscale_8bits(image)

        # Calculate patch coordinates
        height, width = img.shape[:2]
        step = patch_size - patch_overlap

        # Generate all possible patch starting coordinates
        patch_args = []
        index = 0
        for i in range(0, height - patch_size + 1, step):
            for j in range(0, width - patch_size + 1, step):
                args = PatchArgs(
                    i=i,
                    j=j,
                    image=img,
                    patch_size=patch_size,
                    index=index,
                    debug_folder=(
                        str(debug_folder) if debug_folder is not None else None
                    ),
                    no_plate_solving=no_plate_solving,
                    cpulimit_seconds=cpulimit_seconds,
                )
                patch_args.append(args)
                index += 1

        logger.info(f"Plate solving running for {len(patch_args)} patches")

        # Create and solve patches in parallel
        patches = []
        if num_processes == 1 or len(patch_args) == 1:
            # Single process for small number of patches or when requested
            patches = [_create_patch(args) for args in patch_args]
        else:
            # Use multiprocessing
            with mp.Pool(processes=num_processes) as pool:
                patches = list(pool.imap(_create_patch, patch_args))

        # Filter out patches where plate solving failed
        solved_patches = [p for p in patches if p.wcs is not None]
        if (not no_plate_solving) and (not solved_patches):
            raise AstrometryError(b"", b"All patches failed plate solving", 1)

        # Sort patches by their index to ensure consistent ordering
        patches.sort(key=lambda p: p.index)

        logger.info(f"Successfully solved {len(solved_patches)}/{len(patches)} patches")
        return cls(patches, image)
