
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import healpy as hp
import imageio
import numpy as np
from astropy.wcs import WCS
from loguru import logger

from ..conversions import normalize_to_uint8
from .patch_types import Pixel, Size, HEALPixDict, HEALPixNside
from .patch_args import PatchArgs
from ..plate_solving import PlateSolving, AstrometryError, AstrometryFailed
from ..image import ImageData

@dataclass
class Patch:
    index: int
    location: Pixel
    size: Size
    wcs: Optional[WCS]

    def get_image(self, image: np.ndarray) -> np.ndarray:
        # must be robust to 3 channel RGB images
        if len(image.shape)==2:
            return image[
                self.location[0] : self.location[0] + self.size[0], 
                self.location[1] : self.location[1] + self.size[1],
            ]
        return image[
            self.location[0] : self.location[0] + self.size[0], 
            self.location[1] : self.location[1] + self.size[1],
            :
        ]

    def get_healpix_indices(self, nside: HEALPixNside, to_fill_image: np.ndarray)->np.ndarray:
        npix = hp.nside2npix(nside)
        if np.iinfo(np.uint16).max < npix:
            raise ValueError(f"Cannot encode nside {nside} healpix using uint16")
        image_data = self.get_image(to_fill_image)

        def _commented():
            indices = np.zeros(image_data.shape[:2], np.uint16)
            for y in range(image_data.shape[0]):
                for x in range(image_data.shape[1]):
                    ra, dec = self.wcs.wcs_pix2world(x, y, 0)
                    theta = np.radians(90 - dec)
                    phi = np.radians(ra)
                    hp_index = hp.ang2pix(nside, theta, phi)
                indices[y][x] = hp_index
            return indices

        y,x  = np.indices(image_data.shape[:2])
        sky_coords = self.wcs.all_pix2world(x, y, 0)
        # Convert sky coordinates to radians
        ra_rad = np.radians(sky_coords[0])
        dec_rad = np.radians(sky_coords[1])
    
        # Convert to HEALPix indices
        healpix_indices = hp.ang2pix(nside, 
                                np.pi/2 - dec_rad,  # healpy uses theta, phi coordinates
                                ra_rad)
        return healpix_indices
                
    
    def get_healpix_dict(self, nside: HEALPixNside, image: np.ndarray) -> HEALPixDict:
        npix = hp.nside2npix(nside)
        image_data = self.get_image(image)
        r: HEALPixDict = {}
        indices: set[int] = set()
        duplicates: int = 0
        for y in range(image_data.shape[0]):
            for x in range(image_data.shape[1]):
                pixel_value = image_data[y, x]

                # Convert pixel coordinates to celestial coordinates
                ra, dec = self.wcs.wcs_pix2world(x, y, 0)

                # Convert celestial coordinates to HEALPix index
                theta = np.radians(90 - dec)
                phi = np.radians(ra)
                hp_index = hp.ang2pix(nside, theta, phi)
                r[hp_index] = pixel_value
                if hp_index in indices:
                    duplicates += 1
                indices.add(hp_index)
        logger.warning(f"nside: {nside}: Patch {self.index} has {duplicates} duplicate pixels / {image_data.shape[0]*image_data.shape[1]} total pixels")
        return r    

    def to_jpeg(self, image: np.ndarray, target_dir: Path)->None:
        image_data = self.get_image(image)
        image_data = normalize_to_uint8(image_data)
        target_file = target_dir / f"patch_{self.index}.jpeg"
        logger.info(f"Saving patch {self.index} to {target_file}")
        imageio.imwrite(target_file, image_data)

    def to_fits(self, image: np.ndarray, target_dir: Path)->None:
        data = self.get_image(image)
        shape = data.shape
        hdu = fits.PrimaryHDU(data=data)
        hdu.header['SIMPLE'] = True
        hdu.header['BITPIX'] = 8 if data.dtype==np.uint8 else 16
        if len(shape)==3:
            hdu.header['NAXIS'] = 3                     # 3 dimensions: color, height, width
            hdu.header['NAXIS1'] = shape[0]                  # Width
            hdu.header['NAXIS2'] = shape[1]                  # Height
            hdu.header['NAXIS3'] = shape[2]                    # Color channels (RGB)
            hdu.header['CTYPE3'] = 'COLOR'              # Dimension 3 represents colors
            hdu.header['CNAME3'] = ['RED', 'GREEN', 'BLUE']
        else:
            hdu.header['NAXIS'] = 2                     # 2 dimensions: height, width
            hdu.header['NAXIS1'] = shape[0]                  # Width
            hdu.header['NAXIS2'] = shape[1]                  # Height
        target_file = target_dir / f"patch_{self.index}.fits"
        logger.info(f"Saving patch {self.index} (shape:{shape}, dtype: {data.dtype}) to {target_file}")
        hdu.writeto(target_file, output_verify='exception', overwrite=True)

    def to_wcs(self, target_dir: Path) -> None:
        """
        Save the WCS information to a FITS file following the FITS WCS standard.
        
        Args:
            target_dir: Directory where to save the WCS file
            
        Raises:
            ValueError: If the patch doesn't have a valid WCS solution
        """
        if self.wcs is None:
            logger.info(f"Patch {self.index} doesn't have a valid WCS solution, skipping")
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"patch_{self.index}.wcs"
        hdu = fits.PrimaryHDU()
        hdu.header.extend(self.wcs.to_header(), update=True)
        logger.info(f"Saving patch {self.index} WCS to {target_file}")
        hdu.writeto(target_file, overwrite=True)

    @classmethod
    def create(cls, patch_args: PatchArgs) -> "Patch":
        """Helper function to create and solve a single patch.

        This is a module-level function to support multiprocessing.

        Args:
            patch_args: Named tuple containing patch creation arguments

        Returns:
            Patch object with WCS solution if successful, None otherwise
        """
        
        logger.info(
            f"Processing patch {patch_args.index}"
        )

        image_data: ImageData = patch_args.get_image()

        if patch_args.no_plate_solving:
            logger.info(f"Plate solving disabled, skipping")
            return Patch(
                location=patch_args.location,
                size=patch_args.size,
                index=patch_args.index,
                wcs=None,
            )

        try:
            # Solve the plate for this patch
            cpulimit = (
                0 if patch_args.cpulimit_seconds is None else patch_args.cpulimit_seconds
            )

            wcs = PlateSolving.from_numpy(patch_args.label, image_data, cpulimit, Path(patch_args.working_dir))

            logger.info(f"Successfully solved patch {patch_args.index}")
            return Patch(
                location=patch_args.location,
                size=patch_args.size,
                index=patch_args.index,
                wcs=wcs,
            )
        except (AstrometryError, AstrometryFailed) as e:
            logger.warning(
                f"Failed to solve patch {patch_args.index} at "
                f"({patch_args.location}): {str(e)}"
            )
            # Return a patch without WCS if solving fails
            return Patch(
                location=patch_args.location,
                size=patch_args.size,
                index=patch_args.index,
                wcs=None,
            )
