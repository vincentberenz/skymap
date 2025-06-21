from skymapper.patches import PatchedImage, HealpixPatchedImage
from skymapper.healpix import HealpixImage


INPUT_PATCHES = "images/nightskycam3_2025_04_05_03_54_30.pkl.gz"
OUTPUT_HEALPIX = "images/nightskycam3_2025_04_05_03_54_30.healpix"

if __name__ == "__main__":
    
    patches = PatchedImage.load(INPUT_PATCHES)
    healpix: HealpixPatchedImage = patches.to_healpix(128)
    healpix_image = HealpixImage(healpix, 128)
    data: np.ndarray[tuple[int], np.uint8] = healpix_image.combine()
