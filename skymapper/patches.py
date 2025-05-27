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
    patch_size: int = 512,
    overlap: float = 0.5,
    min_stars: int = 5,
    star_threshold: float = 100,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Extract patches from an image for star detection

    Parameters:
    -----------
    image : np.ndarray
        Input image (can be grayscale or color)
    patch_size : int
        Size of each patch
    overlap : float
        Overlap ratio between patches (0.0 to 1.0)
    min_stars : int
        Minimum stars required in a patch
    star_threshold : float
        Threshold for star detection

    Returns:
    --------
    List of tuples containing (patch, x_offset, y_offset)
    """
    patches = []

    # Handle both grayscale and color images
    if len(image.shape) == 3:  # Color image (height, width, channels)
        h, w = image.shape[:2]
    else:  # Grayscale image (height, width)
        h, w = image.shape

    step = int(patch_size * (1 - overlap))

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Calculate patch bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + patch_size)
            y2 = min(h, y + patch_size)

            # Extract patch
            patch = image[y1:y2, x1:x2]

            # Convert to grayscale if needed for star detection
            if len(patch.shape) == 3:
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            else:
                gray_patch = patch

            # adding code for displaying the patch in a window, for debug
            # cv2.imshow("Patch", patch)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Detect stars in patch
            _, binary = cv2.threshold(
                gray_patch, star_threshold, 255, cv2.THRESH_BINARY
            )
            contours, _ = cv2.findContours(
                binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            # Count stars
            star_count = sum(1 for contour in contours if cv2.contourArea(contour) > 10)

            if star_count >= min_stars:
                patches.append((patch, x1, y1))

    return patches


def extract_patches_from_file(
    img_path: str,
    patch_size: int,
    patch_overlap: int,
    output: str,
    min_stars: int = 5,
    star_threshold: int = 20,
) -> int:
    img_ = load_image(img_path)
    img = to_8bits(img_)
    logger.info(
        f"Extracting patches from {Path(img_path).stem} ({img.shape}, {img.dtype})"
    )
    # Extract patches with proper parameters
    image_patches: list[tuple[np.ndarray, int, int]] = extract_image_patches(
        img,
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
