from pathlib import Path
from skymapper.patches import PatchedImage


TARGET_FOLDER = "images/jpg/"
INPUT_FILE = Path("images/nightskycam3_2025_04_05_03_54_30.pkl.gz")
STRETCH = True

if __name__ == "__main__":
    
    patches = PatchedImage.load(INPUT_FILE)
    patches.to_jpeg(Path(TARGET_FOLDER), STRETCH)
