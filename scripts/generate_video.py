import os
import cv2
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from loguru import logger
import numpy as np
from tqdm import tqdm

# Set the number of worker processes (use all available CPUs - 1)
NUM_WORKERS = max(1, mp.cpu_count() - 1)

BASE_FOLDER = Path("/home/vberenz/Workspaces/skymap-data/june_19th_2025/ns3/")
IN_FOLDER = BASE_FOLDER / "healpix/"
OUT_PATH = BASE_FOLDER / "video" / "healpix-indices.mp4"
SCALE_FACTOR = 0.5
FPS = 2

def process_single_image(args: Tuple[Path, float]) -> Optional[Tuple[int, np.ndarray]]:
    """
    Process a single image file with error handling.
    
    Args:
        args: Tuple of (image_path, scale_factor)
        
    Returns:
        Tuple of (timestamp, resized_image) or None if processing failed
    """
    image_path, scale_factor = args
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return None
            
        # Handle different image depths
        if img.dtype == np.uint16:
            if img.max() > 255:
                img = (img / 256).astype(np.uint8)
        
        # Convert to RGB if single channel
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
        # Calculate new dimensions
        new_size = (int(img.shape[1] * scale_factor),
                   int(img.shape[0] * scale_factor))
        
        # Resize using INTER_AREA for downscaling
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return (int(parse_nightskycam_timestamp(image_path.name).timestamp()), resized)
        
    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {str(e)}")
        return None

def process_images_parallel(image_paths: List[Path], scale_factor: float = 0.5) -> List[np.ndarray]:
    """
    Process multiple images in parallel using multiprocessing.
    
    Args:
        image_paths: List of image paths to process
        scale_factor: Scale factor for resizing
        
    Returns:
        List of image arrays in chronological order
    """
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Create tasks
        tasks = [(img_path, scale_factor) for img_path in image_paths]
        
        # Process images in parallel with progress bar
        with tqdm(total=len(tasks), desc="Processing frames") as pbar:
            futures = {executor.submit(process_single_image, task): i 
                     for i, task in enumerate(tasks)}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    # Sort results by timestamp to maintain chronological order
    results.sort(key=lambda x: x[0])
    return [img for _, img in results]

def create_video_from_frames(frames: List[np.ndarray], output_path: str = 'output.mp4', 
                           fps: int = 2, codec: str = 'mp4v') -> None:
    """
    Create a video from a list of frames.
    
    Args:
        frames: List of image arrays
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec to use (default: 'mp4v')
    """
    if not frames:
        raise ValueError("No frames to create video")
    
    # Get frame dimensions from the first frame
    height, width = frames[0].shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame in tqdm(frames, desc="Writing video"):
        out.write(frame)
    
    # Release everything
    out.release()
    logger.info(f"Video saved to: {output_path}")

def parse_nightskycam_timestamp(filename):
    """
    Extract timestamp from nightskycam3 filename.
    
    Args:
        filename: Filename in format: nightskycam3_<YYYY>_<MM>_<DD>_<H>_<M>_<S>.pkl.overlayed.tiff
        
    Returns:
        datetime object representing the timestamp
    """
    # Remove prefix and suffix
    timestamp_str = filename.replace('nightskycam3_', '').replace('.pkl.overlayed.tiff', '')
    
    # Parse into datetime
    try:
        return datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M_%S')
    except ValueError:
        raise ValueError(f"Invalid timestamp format in filename: {filename}")

def process_images_in_memory(input_folder: str, output_path: str, 
                           scale_factor: float = 0.5, fps: int = 2) -> None:
    """
    Process all nightskycam3 TIFF images in a folder and create a video using multiprocessing.
    Files are processed in chronological order based on their timestamps.
    
    Args:
        input_folder: Path to the folder containing images
        output_path: Output video file path
        scale_factor: Scale factor for resizing images (default: 0.5)
        fps: Frames per second for the output video (default: 2)
    """
    folder = Path(input_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {input_folder}")
    
    # Get all nightskycam3 TIFF files
    image_files = list(folder.glob('nightskycam3_*.pkl.overlayed.tiff'))
    if not image_files:
        raise FileNotFoundError(f"No nightskycam3 TIFF files found in {input_folder}")
    
    logger.info(f"Found {len(image_files)} nightskycam3 TIFF files in {input_folder}")
    logger.info(f"Using {NUM_WORKERS} worker processes")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process images in parallel
    frames = process_images_parallel(image_files, scale_factor)
    
    if not frames:
        raise RuntimeError("No frames were successfully processed")
    
    # Create video
    create_video_from_frames(frames, str(output_path), fps)

if __name__ == "__main__":
    process_images_in_memory(IN_FOLDER, output_path=OUT_PATH, scale_factor=SCALE_FACTOR, fps=FPS)