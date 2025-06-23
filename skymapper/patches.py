import gzip
import multiprocessing as mp
import os
import pickle
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NamedTuple, NewType, Optional, Tuple
import healpy as hp
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
        patch_size: Tuple of (height, width) for the patch
        index: Unique identifier for the patch
        debug_folder: Optional folder for debug output
        no_plate_solving: If True, skip plate solving
        cpulimit_seconds: CPU time limit for plate solving
    """

    i: int
    j: int
    image: np.ndarray
    patch_size: tuple[int, int]
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
class HealpixPatch:
    data: np.ndarray[tuple[int], np.uint8]
    mask: np.ndarray[tuple[int], np.bool]

@dataclass
class Patch:
    index: int
    location: Pixel
    size: tuple[int, int]
    wcs: Optional[WCS]

    def get_image(self, image: np.ndarray) -> np.ndarray:
        return image[
            self.location[0] : self.location[0] + self.size[0], 
            self.location[1] : self.location[1] + self.size[1]
        ]

    def get_resolution(self) -> float:
        try:
            # Try to get the CD matrix
            cd_matrix = self.wcs.wcs.cd
        except AttributeError:
            # If CD is not available, try to get the PC matrix
            cd_matrix = self.wcs.wcs.pc
        
        # If neither CD nor PC is available, assume a default resolution
        if cd_matrix is None:
            logger.warning("WCS object lacks CD matrix. Using default resolution.")
            # This is a fallback; adjust as needed for your specific case
            return 1.0  # Default resolution in arcseconds
    
        # Convert degrees to arcseconds
        return cd_matrix[0][0] * 3600  # Assuming x-pixel size


    def get_extents(self) -> tuple[float, float, float, float]:
        try:
            crpix = self.wcs.wcs.crpix
            crval = self.wcs.wcs.crval
            cd = self.wcs.wcs.cd
        except AttributeError as e:
            logger.warning(f"WCS object lacks some attributes. Error: {e}")
            # Fallback method if we don't have CD matrix
            ra_min = crpix[0] - 10 * np.sqrt(crpix[0]**2 + crpix[1]**2)
            ra_max = crpix[0] + 10 * np.sqrt(crpix[0]**2 + crpix[1]**2)
            dec_min = crpix[1] - 10 * abs(crpix[0])
            dec_max = crpix[1] + 10 * abs(crpix[0])
        else:
            # Using the standard method when CD is available
            ra_min = crpix[0] - cd[0][0] * crval[0]
            ra_max = crpix[0] + cd[0][0] * (crval[0] + cd[1][1] * crval[1])
            dec_min = crpix[1] - cd[1][1] * crval[1]
            dec_max = crpix[1] + cd[1][1] * (crval[1] + cd[0][0] * crval[0])

        return ra_min, ra_max, dec_min, dec_max
    
    def update_healpix(
        self, image: np.ndarray, nside: int, map: np.ndarray[tuple[int], np.uint8], 
    ) -> np.ndarray[tuple[int], np.uint8]:
       

        npix = hp.nside2npix(nside)

        image_data = self.get_image(image)

        logger.info(f"updating healpix map for patch {self.index}. Original image: {image.shape}, patch image: {image_data.shape}")

        updated_indices: set[int] = set()

        for y in range(image_data.shape[0]):
            for x in range(image_data.shape[1]):
                # Get the pixel value
                pixel_value = image_data[y, x]

                # Convert pixel coordinates to celestial coordinates
                ra, dec = self.wcs.wcs_pix2world(x, y, 0)

                # Convert celestial coordinates to HEALPix index
                theta = np.radians(90 - dec)
                phi = np.radians(ra)
                hp_index = hp.ang2pix(nside, theta, phi)

                # Accumulate the pixel value into the HEALPix map
                map[hp_index] = pixel_value
                updated_indices.add(hp_index)

        logger.info(f"Updated {len(updated_indices)} indices out of {image_data.shape[0]*image_data.shape[1]} (npix={npix})")

        return map

    def to_healpix_not_vectorized(self, image: np.ndarray, nside: int) -> HealpixPatch:

        npix = healpy.nside2npix(nside)
        healpix_map = np.zeros(npix, dtype=np.uint8)
        healpix_mask = np.zeros(npix, dtype=np.bool)

        image_data = image[
            self.location[0] : self.location[0] + self.size[0], 
            self.location[1] : self.location[1] + self.size[1]
        ]

        for y in range(image_data.shape[0]):
            for x in range(image_data.shape[1]):
                # Get the pixel value
                pixel_value = image_data[y, x]

                # Convert pixel coordinates to celestial coordinates
                ra, dec = self.wcs.wcs_pix2world(x, y, 0)

                # Convert celestial coordinates to HEALPix index
                theta = np.radians(90 - dec)
                phi = np.radians(ra)
                hp_index = healpy.ang2pix(nside, theta, phi)

                # Accumulate the pixel value into the HEALPix map
                healpix_map[hp_index] += pixel_value
                healpix_mask[hp_index] = True

        return HealpixPatch(data=healpix_map, mask=healpix_mask)    

    def to_healpix(self, image: np.ndarray, nside: int) -> HealpixPatch:
        npix = hp.nside2npix(nside)
        healpix_map = np.zeros(npix, dtype=np.uint8)
        healpix_mask = np.zeros(npix, dtype=bool)
        
        # Extract the relevant portion of the image
        image_data = image[
            self.location[0]:self.location[0] + self.size[0],
            self.location[1]:self.location[1] + self.size[1]
        ]
        
        # Generate pixel coordinates
        y_indices, x_indices = np.indices(image_data.shape)
        
        # Convert pixel coordinates to celestial coordinates
        ra, dec = self.wcs.wcs_pix2world(x_indices, y_indices, 0)
        
        # Convert celestial coordinates to HEALPix indices
        theta = np.radians(90 - dec)
        phi = np.radians(ra)
        hp_indices = hp.ang2pix(nside, theta, phi)
        
        # Flatten the arrays to ensure compatibility
        hp_indices = hp_indices.flatten()
        image_data_flat = image_data.flatten()
        
        # Accumulate the pixel values into the HEALPix map
        np.add.at(healpix_map, hp_indices, image_data_flat)
        
        # Create the mask
        healpix_mask[hp_indices] = True
        
        return HealpixPatch(data=healpix_map, mask=healpix_mask)

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


def calculate_output_shape(patches: list[Patch])->tuple[int,int]:

    total_ra_min = float('inf')
    total_ra_max = float('-inf')
    total_dec_min = float('inf')
    total_dec_max = float('-inf')

    for patch in patches:
        ra_min, ra_max, dec_min, dec_max = patch.get_extents()
        
        total_ra_min = min(total_ra_min, ra_min)
        total_ra_max = max(total_ra_max, ra_max)
        total_dec_min = min(total_dec_min, dec_min)
        total_dec_max = max(total_dec_max, dec_max)

    # Calculate the total extent in RA and Dec
    total_extent_ra = total_ra_max - total_ra_min
    total_extent_dec = total_dec_max - total_dec_min

    # Extract resolution from the first image's WCS
    current_resolution = patches[0].get_resolution()
    logger.info(f"Resolution: {current_resolution}")

    # Use the extracted resolution for our calculation
    num_pixels_ra = int(np.ceil(total_extent_ra / current_resolution))
    num_pixels_dec = int(np.ceil(total_extent_dec / current_resolution))

    return num_pixels_ra, num_pixels_dec


def combine_images(patches: list[Patch], full_image: np.ndarray)->tuple[np.ndarray, WCS]:
    reference_wcs = patches[0].wcs
    output_shape = calculate_output_shape(patches)
    combined_images = np.zeros(output_shape, dtype=full_image.dtype)

    for patch in patches:
        image = patch.get_image(full_image)
        reprojected_image, _ = reproject_interp((image, patch.wcs), reference_wcs, shape_out=output_shape)
        combined_images += reprojected_image

    return combined_images, reference_wcs
    


def _create_patch(patch_args: PatchArgs) -> Patch:
    """Helper function to create and solve a single patch.

    This is a module-level function to support multiprocessing.

    Args:
        patch_args: Named tuple containing patch creation arguments

    Returns:
        Patch object with WCS solution if successful, None otherwise
    """
    patch_height, patch_width = patch_args.patch_size
    patch_data = patch_args.image[
        patch_args.i : patch_args.i + patch_height,
        patch_args.j : patch_args.j + patch_width,
    ]

    if patch_args.debug_folder is not None:
        p = Path(patch_args.debug_folder)
        p.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(p / f"patch_{patch_args.index:04d}.tiff", patch_data)

    logger.info(
        f"Processing patch {patch_args.index} / {patch_data.shape} / {patch_data.dtype}"
    )

    if patch_args.no_plate_solving:
        logger.info(f"Plate solving disabled, skipping")
        return Patch(
            location=(patch_args.i, patch_args.j),
            size=patch_args.patch_size,
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
            size=patch_args.patch_size,
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
            size=patch_args.patch_size,
            index=patch_args.index,
            wcs=None,
        )


@dataclass
class HealpixPatchedImage:
    data: np.ndarray[tuple[int, int], np.uint8]
    mask: np.ndarray[tuple[int, int], np.bool]


@dataclass
class PatchedImage:
    patches: list[Patch]
    image: np.ndarray

    def to_healpix(self, nside: int, patches: Optional[tuple[int,...]] = None) -> np.ndarray[tuple[int], np.float32]:

        # returns a 2d numpy array, one line per patch (calling to_healpix on each patch)

        if patches is not None:
            _patches = [p for p in self.patches if p.index in patches]
        else:
            _patches = [p for p in self.patches if p.wcs is not None]

        # image should be uint8 and grayscale
        img = to_grayscale_8bits(self.image)

        npix = hp.nside2npix(nside)

        map = np.full(npix, hp.UNSEEN, dtype=np.uint8)

        for i, patch in enumerate(_patches):
            logger.info(f"Processing patch {i+1} / {len(_patches)} (patch: {patch.index})")
            map = patch.update_healpix(img, nside, map)

        return map


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
        num_processes: Optional[int] = get_num_processes(),
        debug_folder: Optional[Path] = None,
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
            image=image,
            patch_size=patch_size,
            num_processes=num_processes,
            debug_folder=debug_folder,
            no_plate_solving=no_plate_solving,
            cpulimit_seconds=cpulimit_seconds,
        )

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        patch_size: tuple[int, int],
        num_processes: Optional[int] = get_num_processes(),
        debug_folder: Optional[Path] = None,
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
        if num_processes is None:
            num_processes = 1

        img = to_grayscale_8bits(image)
        patch_height, patch_width = patch_size
        height, width = img.shape[:2]
        
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
        logger.debug(f"Image size: {img.shape}")
        logger.debug(f"Number of patches: {num_patches_y}x{num_patches_x}")
        logger.debug(f"Computed overlap: y={overlap_y}, x={overlap_x}")
        logger.debug(f"Step sizes: y={step_y}, x={step_x}")

        # Generate all possible patch coordinates
        patch_args = []
        index = 0
        for y in range(0, height - patch_height + 1, step_y):
            for x in range(0, width - patch_width + 1, step_x):
                # Calculate actual patch dimensions (may be smaller at edges)
                actual_height = min(patch_height, height - y)
                actual_width = min(patch_width, width - x)
                
                patch_args.append(
                    PatchArgs(
                        i=y,
                        j=x,
                        image=img,
                        patch_size=(actual_height, actual_width),
                        index=index,
                        debug_folder=str(debug_folder) if debug_folder else None,
                        no_plate_solving=no_plate_solving,
                        cpulimit_seconds=cpulimit_seconds,
                    )
                )
                index += 1
                
                # If we've reached the end of the row, break to avoid extra patches
                if x + step_x >= width - patch_width + 1:
                    break
            # If we've reached the end of the column, break to avoid extra patches
            if y + step_y >= height - patch_height + 1:
                break

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
