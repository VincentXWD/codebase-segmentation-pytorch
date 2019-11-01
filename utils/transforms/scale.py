import random
import numbers
import collections
import math
import cv2
import numpy as np

from .crop import RandomCrop

from PIL import Image


class FreeScale(object):

  def __init__(self, size):
    self.size = tuple(reversed(size))  # size: (h, w)

  def __call__(self, img, mask):
    assert img.size == mask.size
    return (img.resize(self.size,
                       Image.BILINEAR), mask.resize(self.size, Image.NEAREST))


class RandomScale(object):

  def __init__(self, scale, aspect_ratio=None):
    assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
    if isinstance(scale, collections.Iterable) and len(scale) == 2 \
        and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
        and 0 < scale[0] < scale[1]:
      self.scale = scale
    else:
      raise (RuntimeError("RandomScale() scale param error.\n"))
    if aspect_ratio is None:
      self.aspect_ratio = aspect_ratio
    elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
        and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
        and 0 < aspect_ratio[0] < aspect_ratio[1]:
      self.aspect_ratio = aspect_ratio
    else:
      raise (RuntimeError("RandomScale() aspect_ratio param error.\n"))

  def __call__(self, image, label):
    image = np.asarray(image)
    label = np.asarray(label)
    temp_scale = self.scale[0] + (self.scale[1] -
                                  self.scale[0]) * random.random()
    temp_aspect_ratio = 1.0
    if self.aspect_ratio is not None:
      temp_aspect_ratio = self.aspect_ratio[0] + (
          self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
      temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
    scale_factor_x = temp_scale * temp_aspect_ratio
    scale_factor_y = temp_scale / temp_aspect_ratio
    image = cv2.resize(
        image,
        None,
        fx=scale_factor_x,
        fy=scale_factor_y,
        interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(
        label,
        None,
        fx=scale_factor_x,
        fy=scale_factor_y,
        interpolation=cv2.INTER_NEAREST)
    return image, label


class RandomSized(object):

  def __init__(self, size):
    self.size = size
    self.scale = RandomScale(self.size)
    self.crop = RandomCrop(self.size)

  def __call__(self, img, mask):
    assert img.size == mask.size

    w = int(random.uniform(0.5, 2) * img.size[0])
    h = int(random.uniform(0.5, 2) * img.size[1])

    img, mask = (img.resize((w, h),
                            Image.BILINEAR), mask.resize((w, h), Image.NEAREST))

    return self.crop(*self.scale(img, mask))


class Resize(object):

  def __init__(self, size):
    assert (isinstance(size, collections.Iterable) and len(size) == 2)
    self.size = size

  def __call__(self, image, label):
    image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
    return image, label
