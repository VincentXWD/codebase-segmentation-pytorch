import torch
import numpy as np

from PIL import Image

from .adjust import AdjustBrightness, AdjustContrast, AdjustGamma, AdjustHue, AdjustSaturation
from .crop import RandomCrop
from .flip import RandomHorizontallyFlip, RandomVerticallyFlip
from .scale import RandomScale, FreeScale, RandomSized
from .rotate import RandomRotate
from .blur import RandomGaussianBlur

__all__ = [
    'Compose', 'ToTensor', 'Normalize', 'RandomCrop', 'AdjustBrightness',
    'AdjustContrast', 'AdjustGamma', 'AdjustHue', 'AdjustSaturation',
    'RandomHorizontallyFlip', 'RandomVerticallyFlip', 'RandomScale',
    'FreeScale', 'RandomSized', 'RandomRotate', 'RandomGaussianBlur'
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


class ToTensor(object):

  def __call__(self, image, label):
    if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
      raise (RuntimeError("ToTensor() only handle np.ndarray"
                          "[eg: data readed by cv2.imread()].\n"))
    if len(image.shape) > 3 or len(image.shape) < 2:
      raise (RuntimeError(
          "ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
    if len(image.shape) == 2:
      image = np.expand_dims(image, axis=2)
    if not len(label.shape) == 2:
      raise (RuntimeError(
          "ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

    image = torch.from_numpy(image.transpose((2, 0, 1)))
    if not isinstance(image, torch.FloatTensor):
      image = image.float()
    label = torch.from_numpy(label)
    if not isinstance(label, torch.LongTensor):
      label = label.long()
    return image, label


class Normalize(object):

  def __init__(self, mean, std=None):
    if std is None:
      assert len(mean) > 0
    else:
      assert len(mean) == len(std)
    self.mean = mean
    self.std = std

  def __call__(self, image, label):
    if self.std is None:
      for t, m in zip(image, self.mean):
        t.sub_(m)
    else:
      for t, m, s in zip(image, self.mean, self.std):
        t.sub_(m).div_(s)
    return image, label


class RGB2BGR(object):

  def __call__(self, image, label):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, label


class BGR2RGB(object):

  def __call__(self, image, label):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, label
