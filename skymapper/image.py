from dataclasses import dataclass
import numpy as np
from pathlib import Path
import imageio
from .conversions import stretch, normalize_to_float32, normalize_to_uint8, to_grayscale
from .patches.patch_types import Pixel, Size

@dataclass
class ImageConfig:
    stretch: bool
    grayscale: bool
    dtype: np.dtype

class ImageData:

    def __init__(self, image: np.ndarray, image_config: ImageConfig)->None:
        self.image = image
        self.config = image_config

        if image_config.stretch:
            self.image = stretch(self.image)

        if self.image.dtype != image_config.dtype:
            if image_config.dtype == np.uint8:
                self.image = normalize_to_uint8(self.image)
            elif image_config.dtype == np.float32:
                self.image = normalize_to_float32(self.image)
            else:
                raise ValueError(f"Unsupported dtype: {self.dtype}")

        if image_config.grayscale:
            self.image = to_grayscale(self.image)

    def save(self, folder: Path, filename: str)->Path:

        folder.mkdir(parents=True, exist_ok=True)
        filename = f"{filename}_{self.image.shape[0]}x{self.image.shape[1]}_{self.image.dtype.name}"
        if self.config.stretch:
            filename += "_stretched"
        if self.config.grayscale:
            filename += "_grayscaled"
        if self.image.dtype == np.uint8:
            format = "png"
        else:
            format = "tiff"
        total_path = folder / f"{filename}.{format}"
        imageio.imwrite(total_path, self.image, format=format)
        return total_path
        
    def get(self, location: Pixel, size: Size)->"ImageData":
        if len(self.image.shape)==2:
            image = self.image[
                location[0] : location[0] + size[0], 
                location[1] : location[1] + size[1],
            ]
        else:
            image = self.image[
                location[0] : location[0] + size[0], 
                location[1] : location[1] + size[1],
                :
            ]
        return self.__class__(image, self.config)
