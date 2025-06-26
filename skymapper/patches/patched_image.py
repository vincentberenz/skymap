from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import healpy as hp
import imageio
import numpy as np
from astropy.wcs import WCS
from loguru import logger
from multipledispatch import dispatch
import multiprocessing as mp
import pickle
import gzip

from ..conversions import normalize_to_uint8, stretch, to_grayscale_8bits
from .patch import Patch
from .patch_args import PatchArgs
from .patch_coords import PatchCoords
from .patch_types import Pixel, Size, Shape
from ..healpix import HEALPixDict, HEALPixNside, save_healpix_dict, HEALPixDictRecord
from ..plate_solving import AstrometryError, AstrometryFailed
from ..image import ImageConfig, ImageData

def get_num_processes() -> int:
    """Get the number of processes to use for parallel processing.
    
    Returns:
        int: Number of processes to use (total CPUs - 1, minimum 1)
    """
    import os
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, cpu_count - 1)


@dataclass
class PatchedImage:
    patches: list[Patch]
    image: np.ndarray

    def to_jpeg(self, target_dir: Path, perform_stretch: bool)->None:
        target_dir.mkdir(parents=True, exist_ok=True)
        image = self.image
        if perform_stretch:
            image = stretch(image)
        patches = [p for p in self.patches if p.wcs is not None]
        for patch in patches:
            patch.to_jpeg(image,target_dir)
            patch.to_wcs(target_dir)

    def get_patch_indices(self)->list[int]:
        return [p.index for p in self.patches if p.wcs is not None]

    def display(
        self, path: Path, border_thickness: int = 2, text_scale: float = 0.8, 
        text_thickness: int = 2, apply_stretch: bool = True
    ) -> None:
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

        logger.info(f"Loading PatchedImage from {path}")

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

    def get_corresponding_patch(self, pixels: Pixel) -> Patch:
        candidates = [patch for patch in self.patches if patch.inside(pixels)]
        if not candidates:
            raise ValueError(f"No patch found for pixels {pixels}")
        return min(candidates, key=lambda patch: patch.distance_to_center(pixels))

    def get_patch(self, index: int) -> Patch:
        return [p for p in self.patches if p.index == index][0]

    @dispatch(int)
    def get_image(self, patch_index: int) -> np.ndarray:
        patch = [p for p in self.patches if p.index == patch_index][0]
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
        patch_size: tuple[int, int],
        working_dir: Path,
        image_config: ImageConfig,
        num_processes: Optional[int] = get_num_processes(),
        no_plate_solving: bool = False,
        cpulimit_seconds: Optional[int] = None,
    ) -> "PatchedImage":
        """Create a PatchedImage from an image file.
        
        Args:
            path: Path to the image file
            patch_size: Tuple of (height, width) for each patch
            num_processes: Number of processes to use for parallel processing
            debug_folder: Optional folder for debug output
            no_plate_solving: If True, skip plate solving
            cpulimit_seconds: CPU time limit for plate solving
            
        Returns:
            PatchedImage instance with automatically computed patch overlap
        """
        image = imageio.imread(path)
        return cls.from_image(
            label=str(path.stem),
            image=image,
            patch_size=patch_size,
            working_dir=working_dir,
            image_config = image_config,
            num_processes=num_processes,
            no_plate_solving=no_plate_solving,
            cpulimit_seconds=cpulimit_seconds,
        )

    @classmethod
    def from_image(
        cls,
        label: str,
        image: np.ndarray,
        patch_size: tuple[int, int],
        working_dir: Path,
        image_config: ImageConfig,
        num_processes: Optional[int] = get_num_processes(),
        no_plate_solving: bool = False,
        cpulimit_seconds: Optional[int] = None,
    ) -> "PatchedImage":
        """
        Create a PatchedImage by dividing the input image into overlapping rectangular patches.
        Uses multiprocessing to speed up patch creation and plate solving.
        The overlap between patches is automatically computed to ensure full image coverage.

        Args:
            image: Input image as a numpy array
            patch_size: Tuple of (height, width) for each patch in pixels
            num_processes: Number of processes to use for parallel processing.
                        If None, uses all available CPU cores.
            debug_folder: Optional folder to save debug information
            no_plate_solving: If True, skip plate solving
            cpulimit_seconds: CPU time limit for plate solving

        Returns:
            PatchedImage instance containing all patches with WCS solutions

        Raises:
            AstrometryError: If plate solving fails for any patch
        """

        image_data = ImageData(image, image_config)

        if num_processes is None:
            num_processes = 1
       
        patches: list[PatchCoords] = PatchCoords.create(image.shape, patch_size)
        patch_args: list[PatchArgs] = []

        for index,patch in enumerate(patches):
            patch_args.append(
                PatchArgs(
                    label=f"{label}_{index}",
                    location=(patch.location[0],patch.location[1]),
                    size=(patch.size[0], patch.size[1]),
                    image_data=image_data,
                    index=index,
                    no_plate_solving=no_plate_solving,
                    cpulimit_seconds=cpulimit_seconds,
                    working_dir=str(working_dir)
                )
            )

        logger.info(f"Plate solving running for {len(patch_args)} patches")

        # Create and solve patches in parallel
        patches = []
        if num_processes == 1 or len(patch_args) == 1:
            # Single process for small number of patches or when requested
            patches = [Patch.create(args) for args in patch_args]
        else:
            # Use multiprocessing
            with mp.Pool(processes=num_processes) as pool:
                patches = list(pool.imap(Patch.create, patch_args))

        # Filter out patches where plate solving failed
        solved_patches = [p for p in patches if p.wcs is not None]
        if (not no_plate_solving) and (not solved_patches):
            raise AstrometryError(b"", b"All patches failed plate solving", 1)

        # Sort patches by their index to ensure consistent ordering
        patches.sort(key=lambda p: p.index)

        logger.info(f"Successfully solved {len(solved_patches)}/{len(patches)} patches")
        return cls(patches, image)

    @classmethod
    def from_folder(
        cls,
        folder: Path,
        patch_size: tuple[int, int],
        working_dir: Path,
        output_dir: Path,
        image_config: ImageConfig,
        num_processes: Optional[int] = get_num_processes(),
        no_plate_solving: bool = False,
        cpulimit_seconds: Optional[int] = None,
        file_extention: str = "tiff"
    ) ->None:

        # listing all the image files from folder:
        image_files = [f for f in folder.iterdir() if f.suffix == f".{file_extention}"]

        # calling from_file for each image file
        for index, image_file in enumerate(image_files):

            logger.info(f"Running patch generation and plate solving for {image_file.stem} ({index+1}/{len(image_files)})")

            patched_image = cls.from_file(
                image_file, patch_size, working_dir, image_config, 
                num_processes, no_plate_solving, cpulimit_seconds
            )

            output_path = output_dir / f"{image_file.stem}.pkl.gz"
            logger.info(f"Saving processed image to: {output_path}")
            patched_image.dump(output_path)

            # Create and save visualization
            vis_path = output_path.with_name(f"{output_path.stem}_visualization.tiff")
            logger.info(f"Creating visualization: {vis_path}")
            patched_image.display(vis_path)

    def _process_patch_healpix(self, args: tuple[Patch, HEALPixNside]) -> HEALPixDict:
        """Helper function to process a single patch's HEALPix data.
        
        Args:
            args: Tuple of (patch, nside)
            
        Returns:
            HEALPixDict containing the patch's HEALPix data
        """
        patch, nside, image = args
        return patch.get_healpix_dict(nside, image)

    def get_healpix_indices(self, nside: HEALPixNside)->np.ndarray:
        npix = hp.nside2npix(nside)
        indices = np.zeros(self.image.shape[:2], np.uint16)
        for patch in self.patches:
            logger.info(f"processing patch {patch.index}")
            if patch.wcs is not None:
                logger.info(f"patch {patch.index} has transforms")
                patch_indices: np.ndarray = patch.get_healpix_indices(nside, self.image)
                indices[
                    patch.location[0]:patch.location[0]+patch.size[0], 
                    patch.location[1]:patch.location[1]+patch.size[1]
                ] = patch_indices
            else:
                indices[
                    patch.location[0]:patch.location[0]+patch.size[0], 
                    patch.location[1]:patch.location[1]+patch.size[1]
                ] = np.zeros((patch.size[0], patch.size[1]), np.uint16)
        return indices

    def get_healpix_record(
        self, nside: HEALPixNside, 
        num_processes: Optional[int] = None, 
        output_path: Optional[Path] = None,
        patch_indices: Optional[list[int]] = None
    ) -> HEALPixDictRecord:
        """Get a dictionary mapping HEALPix indices to pixel values.
        
        Args:
            nside: HEALPix nside parameter
            num_processes: Number of processes to use for parallel processing.
                         If None, uses all available CPU cores.
                         
        Returns:
            Dictionary mapping HEALPix indices to pixel values
        """
        if num_processes is None:
            num_processes = 1

        # Filter out patches without WCS solutions
        if patch_indices is None:
            valid_patches = [p for p in self.patches if p.wcs is not None]
        else:
            valid_patches = [p for p in self.patches if p.wcs is not None and p.index in patch_indices]


        if not valid_patches:
            logger.warning("No patches with valid WCS solutions found")
            return {}
            
        # Prepare arguments for each patch
        args = [(patch, nside, self.image) for patch in valid_patches]
        
        # Use multiprocessing for large numbers of patches
        if len(valid_patches) > 1 and num_processes > 1:
            logger.info(f"Processing {len(valid_patches)} patches with {num_processes} processes")
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(self._process_patch_healpix, args)
        else:
            # Process serially for small numbers of patches or when num_processes=1
            results = [self._process_patch_healpix(arg) for arg in args]
        
        # Combine results from all patches
        healpix_dict: HEALPixDict = {}
        for result in results:
            healpix_dict.update(result)
            
        logger.info(f"Generated HEALPix dictionary with {len(healpix_dict)} entries")

        if output_path is not None:
            save_healpix_dict(nside, healpix_dict, output_path)

        return nside, healpix_dict
