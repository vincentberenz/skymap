"""Sky Mapper - A tool for mapping fisheye sky images to HEALPix coordinates"""

__version__ = "0.1.0"

from .mapper import (
    load_image,
    correct_fisheye_distortion,
    convert_to_healpix,
    process_image
)
from .plate_solver import solve_plate
from .calibration import (
    load_calibration,
    correct_distortion,
    generate_calibration_from_stars
)
from .utils import visualize_healpix_map, plot_healpix_projection

__all__ = [
    'load_image',
    'correct_fisheye_distortion',

    'convert_to_healpix',
    'process_image',
    'load_calibration',
    'correct_distortion',
    'generate_calibration_from_stars',
    'visualize_healpix_map',
    'plot_healpix_projection'
]
