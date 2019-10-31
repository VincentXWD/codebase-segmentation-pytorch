import numpy as np

from PIL import Image

from .adjust import AdjustBrightness, AdjustContrast, AdjustGamma, AdjustHue, AdjustSaturation
# from .crop import RandomCrop, CenterCrop, RandomSizedCrop
from .crop import RandomCrop, CenterCrop
from .flip import RandomHorizontallyFlip, RandomVerticallyFlip
from .scale import Scale, FreeScale, RandomSized
from .rotate import RandomRotate, RotateDegree

__all__ = [
    'Compose', 'RandomCrop', 'AdjustBrightness', 'AdjustContrast',
    'AdjustGamma', 'AdjustHue', 'AdjustSaturation', 'CenterCrop',
    'RandomHorizontallyFlip', 'RandomVerticallyFlip', 'Scale', 'FreeScale',
    'RandomSized', 'RotateDegree', 'RandomRotate'
]


class Compose(object):

  def __init__(self, augmentations):
    self.augmentations = augmentations
    self.PIL2Numpy = False

  def __call__(self, img, mask):
    if isinstance(img, np.ndarray):
      img = Image.fromarray(img, mode="RGB")
      mask = Image.fromarray(mask, mode="L")
      self.PIL2Numpy = True

    assert img.size == mask.size
    for a in self.augmentations:
      img, mask = a(img, mask)

    if self.PIL2Numpy:
      img, mask = np.array(img), np.array(mask, dtype=np.uint8)

    return img, mask
