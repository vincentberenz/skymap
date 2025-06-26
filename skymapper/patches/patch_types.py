from typing import NewType
import numpy as np

Pixel = NewType("Pixel", tuple[int, int])
Size = NewType("Size", tuple[int, int])
Shape = NewType("Shape", tuple[int, ...])
