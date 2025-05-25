# Sky Mapper

A Python package for mapping fisheye sky images to HEALPix coordinates.

## Features

- Fisheye distortion correction
- Plate solving using astrometry.net
- Conversion to HEALPix coordinates
- Visualization tools
- Camera calibration support

## Installation

You can install the package using pip:

```bash
pip install sky_mapper
```

## Requirements

- Python 3.7+
- numpy>=1.21.0
- astropy>=5.0
- healpy>=1.15.0
- opencv-python>=4.5.0
- astrometry.net>=0.1
- scipy>=1.7.0

## Usage

```python
from skymapper import SkyMapper

# Initialize the mapper
mapper = SkyMapper(
    image_path="path/to/your/image.jpg",
    healpix_nside=256,
    calibration_file="calibration.json"
)

# Process the image
healpix_map = mapper.process_image(api_key="your_astrometry_net_api_key")

# Visualize the result
skymapper.utils.visualize_healpix_map(healpix_map, "output.png")
```

## Camera Calibration

To calibrate your fisheye camera:

```python
from skymapper.calibration import FisheyeCalibrator

# Create calibrator
calibrator = FisheyeCalibrator()

# Calibrate using chessboard pattern
calibrator.calibrate_camera(
    chessboard_size=(6,9),  # Adjust based on your chessboard
    num_samples=20         # Number of calibration samples
)

# Save calibration parameters
calibrator.save_calibration("calibration.json")
```

## License

MIT License
