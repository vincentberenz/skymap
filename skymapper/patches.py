import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .logger import logger
from .utils import load_image, to_8bits


def extract_image_patches(
    image: np.ndarray,
    patch_size: int = 1024,
    overlap: float = 0.5,
    min_patches: int = 10,
    color_space: str = 'bgr',
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Extract patches from an image for processing.

    Parameters:
    -----------
    image : np.ndarray
        Input image (grayscale or color)
    patch_size : int, optional
        Size of each patch in pixels, by default 1024
    overlap : float, optional
        Overlap ratio between patches (0.0 to 1.0), by default 0.5
    min_patches : int, optional
        Minimum number of patches to extract, by default 10
    color_space : str, optional
        Color space of the input image: 'bgr', 'rgb', or 'gray', by default 'bgr'

    Returns:
    --------
    List[Tuple[np.ndarray, int, int]]
        List of tuples containing (grayscale_patch, x_offset, y_offset)
    """
    # Input validation
    if len(image.shape) not in (2, 3):
        raise ValueError(f"Expected 2D (grayscale) or 3D (color) image, got {len(image.shape)}D")
    
    color_space = color_space.lower()
    if color_space not in ('bgr', 'rgb', 'gray'):
        raise ValueError(f"color_space must be 'bgr', 'rgb', or 'gray', got '{color_space}'")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:  # Color image
        if color_space == 'bgr':
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:  # rgb or gray
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:  # Already grayscale
        gray_image = image

    patches = []
    h, w = gray_image.shape
    step = max(1, int(patch_size * (1 - overlap)))
    
    # Ensure we get at least min_patches by adjusting step size if needed
    if step > 1:
        num_patches = ((w - patch_size) // step + 1) * ((h - patch_size) // step + 1)
        if num_patches < min_patches and min_patches > 1:
            step = int((w * h) ** 0.5 / (min_patches ** 0.5))
            step = max(1, min(step, patch_size // 2))  # Ensure some overlap
    
    # Generate patches in a grid
    for y in range(0, max(1, h - patch_size + 1), step):
        for x in range(0, max(1, w - patch_size + 1), step):
            x1, y1 = x, y
            x2, y2 = min(x + patch_size, w), min(y + patch_size, h)
            
            # Skip if patch is too small
            if (x2 - x1) < patch_size // 2 or (y2 - y1) < patch_size // 2:
                continue
                
            patch = gray_image[y1:y2, x1:x2]
            patches.append((patch, x1, y1))
            
            # Stop if we have enough patches
            if len(patches) >= min_patches * 2:  # Get extra for selection
                break
        if len(patches) >= min_patches * 2:
            break
            
    # If we still don't have enough patches, take the whole image as one patch
    if not patches and h > 0 and w > 0:
        patch = cv2.resize(gray_image, (patch_size, patch_size)) if max(h, w) > patch_size else gray_image
        patches = [(patch, 0, 0)]
    
    return patches


def extract_patches_from_file(
    img_path: str,
    patch_size: int,
    patch_overlap: int,
    output: str,
    min_stars: int = 5,
    star_threshold: int = 20,
    color_space: str = 'bgr',
) -> int:
    img_ = load_image(img_path)
    img = to_8bits(img_)
    logger.info(
        f"Extracting patches from {Path(img_path).stem} ({img.shape}, {img.dtype})"
    )
    # Extract patches with proper parameters
    image_patches: list[tuple[np.ndarray, int, int]] = extract_image_patches(
        img,
        color_space=color_space,
        patch_size=patch_size,
        overlap=patch_overlap / patch_size,  # Convert absolute overlap to ratio
        min_stars=min_stars,  # Minimum stars per patch
        star_threshold=star_threshold,  # Default star detection threshold
    )
    logger.info(f"Extracted {len(image_patches)} patches from {img_path}")
    # Save patches to output directory
    for i, (patch, x, y) in enumerate(image_patches):
        patch_path = Path(output) / f"{Path(img_path).stem}_{i:03d}.npz"
        logger.info(f"Saving patch {patch_path.stem} ({patch.shape}, {patch.dtype})")
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(patch_path), patch=patch, x=x, y=y)
    return len(image_patches)


def display_patches(patch_dir: str, delay: int = 0) -> None:
    """
    Display all patches in a directory one by one with their metadata.

    Parameters:
    -----------
    patch_dir : str
        Directory containing patch files (.npz)
    delay : int, optional
        Delay in milliseconds between patches (0 = wait for key press)
    """
    console = Console()
    patch_files = sorted(Path(patch_dir).glob("*.npz"))
    
    if not patch_files:
        logger.warning(f"No .npz files found in {patch_dir}")
        return

    console.print(f"Found {len(patch_files)} patch files in {patch_dir}")
    
    for i, patch_file in enumerate(patch_files, 1):
        try:
            # Load patch data
            data = np.load(patch_file)
            patch = data['patch']
            x = data.get('x', 0)
            y = data.get('y', 0)
            
            # Prepare metadata
            metadata = {
                "File": patch_file.name,
                "Dimensions": f"{patch.shape[1]}x{patch.shape[0]}",
                "Type": str(patch.dtype),
                "Position": f"({x}, {y})",
                "Patch": f"{i}/{len(patch_files)}"
            }
            
            # Display metadata
            console.rule(f"[bold]Patch {i}/{len(patch_files)}")
            for key, value in metadata.items():
                console.print(f"[bold cyan]{key}:[/] {value}")
            
            # Display the patch
            window_name = f"Patch {i}/{len(patch_files)} - {patch_file.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, patch)
            
            # Wait for key press or delay
            key = cv2.waitKey(delay) & 0xFF
            cv2.destroyAllWindows()
            
            # Exit on 'q' or ESC
            if key in (ord('q'), 27):
                console.print("[yellow]Display interrupted by user[/]")
                break
                
        except Exception as e:
            logger.error(f"Error displaying {patch_file}: {str(e)}")
    
    cv2.destroyAllWindows()
    console.print("[green]Finished displaying all patches[/]")
