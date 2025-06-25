import imageio
from skymapper.conversions import stretch
import os

# move all the file `<>_display.jpeg` to `<>_jpeg`

FOLDER = "images/jpeg/"

if __name__ == "__main__":
    for file in os.listdir(FOLDER):
        if file.endswith("_display.jpeg"):
            os.rename(FOLDER + file, FOLDER + file.replace("_display.jpeg", ".jpeg"))