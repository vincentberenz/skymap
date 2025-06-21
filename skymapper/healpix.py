from .patches import HealpixPatchedImage
import numpy as np

class HealpixImage:


    def __init__(self, patches: HealpixPatchedImage, nside: int):
        self._nside = nside
        self._data: np.ndarray[tuple[int, int], np.uint8] = patches.data
        self._mask: np.ndarray[tuple[int, int], np.bool] = patches.mask

    def combine(self) -> np.ndarray[tuple[int], np.uint8]:
        # average the data where the mask is True
        return np.average(self._data, axis=0, weights=self._mask)