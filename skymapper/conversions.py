import numpy as np
from loguru import logger
from auto_stretch.stretch import Stretch


def normalize_to_float32(data: np.ndarray)->np.ndarray:
    logger.debug(f"Normalizing to float32: {data.dtype}")
    min_ = np.iinfo(data.dtype).min
    max_ = np.iinfo(data.dtype).max
    
    # Convert to float32 and scale to preserve dynamic range
    arr_float = data.astype(np.float32)
    arr_float = (arr_float - min_) / (max_ - min_)
    
    return arr_float


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Normalize image data to 8-bit range (0-255)"""
    if data.dtype == np.uint8:
        return data

    # Handle uint16 or other types
    if data.dtype == np.uint16:
        # Scale 16-bit to 8-bit
        logger.debug("Converting uint16 to uint8")
        return (data // 256).astype(np.uint8)

    # For other types, normalize to 0-255 range
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max > data_min:
        data = (data - data_min) * 255.0 / (data_max - data_min)
    return data.astype(np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        logger.debug("Converting color image to grayscale")
        return image.mean(axis=2).astype(np.uint8)
    else:
        raise ValueError("Image must be 2D or 3D")


def to_grayscale_8bits(image: np.ndarray) -> np.ndarray:
    image = normalize_to_uint8(image)
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        image = to_grayscale(image)
        return image
    else:
        raise ValueError("Image must be 2D or 3D")

def stretch(image: np.ndarray) -> np.ndarray:
    def _recast(original_img: np.ndarray, target_image: np.ndarray) -> np.ndarray:
        return (target_image * np.iinfo(original_img.dtype).max).astype(original_img.dtype)

    return _recast(image, Stretch().stretch(image))
