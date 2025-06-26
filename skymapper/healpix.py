from typing import NewType, Optional
import numpy as np
from loguru import logger
from pathlib import Path
import pickle
import healpy as hp
import matplotlib.pyplot as plt


HEALPixIndex = NewType("HEALPixIndex", int)
HEALPixNside = NewType("HEALPixNside", int)
HEALPixDict = NewType("HEALPixDict", dict[HEALPixIndex, np.ndarray])
HEALPixDictRecord = NewType("HEALPixDictRecord", tuple[HEALPixNside, HEALPixDict])

def save_healpix_dict(nside: HEALPixNside, healpix_dict: HEALPixDict, output_path: Path)->None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving HEALPix dictionary to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump((nside, healpix_dict), f)
    logger.info(f"Saved HEALPix dictionary to: {output_path}")


def load_healpix_dict(path: Path)->HEALPixDictRecord:
    logger.info(f"Loading HEALPix dictionary from: {path}")
    with open(path, "rb") as f:
        nside, healpix_dict = pickle.load(f)
    logger.info(f"Loaded HEALPix dictionary from: {path}")
    return HEALPixDictRecord((nside, healpix_dict))

def num_pixels(nside: HEALPixNside)->int:
    return hp.nside2npix(nside)

def healpix_record_info(record: HEALPixDictRecord)->None:
    nside, healpix_dict = record
    npixels = num_pixels(nside)
    logger.info(f"HEALPix dictionary with {len(healpix_dict)} / {npixels} pixels and nside {nside}")

def healpix_index_to_color(nside: HEALPixNside, index: HEALPixIndex) -> np.ndarray:
    """Generate a consistent RGB color for a given HEALPix index.
    
    Uses a hash function to generate consistent colors for the same index.
    The colors are in the range [0, 1] for RGB channels.
    
    Args:
        nside: HEALPix nside parameter
        index: HEALPix pixel index (will be converted to Python int)
        
    Returns:
        numpy.ndarray: RGB color as array of floats in [0, 1]
    """
    # Convert index to Python int to avoid overflow
    index_int = int(index)
    # Use a simple hash function to generate consistent colors
    # This ensures the same index always gets the same color
    hash_val = (index_int * 2654435761) & 0xFFFFFFFF  # Knuth's multiplicative hash
    
    # Extract RGB components using bits from the hash
    r = ((hash_val >> 16) & 0xFF) / 255.0
    g = ((hash_val >> 8) & 0xFF) / 255.0
    b = (hash_val & 0xFF) / 255.0
    
    # Apply a small offset based on nside to ensure different nsides get different colors
    offset = (nside % 10) / 30.0
    r = (r + offset) % 1.0
    g = (g + offset * 1.618) % 1.0  # Golden ratio for better color distribution
    
    return np.array([r, g, b])

def overlay_healpix_indices(nside: HEALPixNside, image_rgb: np.ndarray, indices: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Overlay HEALPix indices on an RGB image with specified transparency.
    
    Args:
        nside: HEALPix nside parameter
        image_rgb: Input RGB image of shape (H, W, 3) with values in [0, max_val]
                  where max_val is 255 for uint8 or 65535 for uint16
        indices: Array of HEALPix indices with shape (H, W)
        alpha: Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)
        
    Returns:
        np.ndarray: Image with HEALPix indices overlaid, same shape and dtype as input
    """
    if image_rgb.dtype not in (np.uint8, np.uint16):
        raise ValueError("Input image must be uint8 or uint16")
        
    # Get max value based on dtype
    max_val = np.iinfo(image_rgb.dtype).max
    
    # Create output image as a copy of the input
    result = image_rgb.copy()
    
    # Create a mask of valid indices (non-zero)
    valid_mask = indices > 0
    
    if not np.any(valid_mask):
        return result
    
    # Get unique indices and their colors
    unique_indices = np.unique(indices[valid_mask])
    
    # Pre-compute colors for all unique indices
    color_map = {}
    for idx in unique_indices:
        if idx > 0:  # Skip background
            color_map[idx] = (np.array(healpix_index_to_color(nside, idx)) * max_val).astype(image_rgb.dtype)
    
    # Apply colors to valid pixels
    for idx, color in color_map.items():
        mask = (indices == idx)
        result[mask] = result[mask] * (1 - alpha) + color * alpha
    
    return result.astype(image_rgb.dtype)


def hammer_plot(record: HEALPixDictRecord, path: Optional[Path] = None) -> None:

    # Convert uint16 RGB to float [0,1] for proper display
    def normalize(rgb_array):
        return rgb_array.astype(np.float32) / float(np.iinfo(rgb_array.dtype).max)

    nside, index_color_dict = record
    
    # Prepare data
    logger.info("Normalizing to float32")
    indices = np.array(list(index_color_dict.keys()))
    colors = np.stack([
        normalize(index_color_dict[i])
        for i in indices
    ])
    
    # Get coordinates for our points
    logger.info("Getting coordinates")
    theta, phi = hp.pix2ang(nside, indices)

    theta = np.degrees(theta)
    phi = np.degrees(phi)
    
    def info(label, a):
        print(label, "shape", a.shape, "dtype", a.dtype, "max", np.max(a), "min", np.min(a))
    info("theta",theta)
    info("phi", phi)
    info("colors", colors)
    
    # Create figure
    logger.info("Plotting (hammer)")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='hammer')
    
    # Plot colored points
    scatter = ax.scatter(phi, 90 - theta, c=colors,
                        transform=plt.Axes.get_transform(ax),
                        s=50)
    plt.title('HEALPix RGB Data')

    if path is not None:
        logger.info(f"Saving plot to {path}")
        plt.savefig(str(path), bbox_inches='tight', dpi=300)
        plt.close()  # Clean up memory
    else:
        plt.show()
