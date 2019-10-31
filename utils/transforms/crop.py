import math
import random
import numbers

from PIL import Image, ImageOps


class RandomCrop(object):

  def __init__(self, size, padding=0):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size
    self.padding = padding

  def __call__(self, img, mask):
    if self.padding > 0:
      img = ImageOps.expand(img, border=self.padding, fill=0)
      mask = ImageOps.expand(mask, border=self.padding, fill=0)

    assert img.size == mask.size
    w, h = img.size
    th, tw = self.size
    if w == tw and h == th:
      return img, mask
    if w < tw or h < th:
      return (img.resize((tw, th),
                         Image.BILINEAR), mask.resize((tw, th), Image.NEAREST))

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return (img.crop(
        (x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))


class CenterCrop(object):

  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, img, mask):
    assert img.size == mask.size
    w, h = img.size
    th, tw = self.size
    x1 = int(round((w - tw) / 2.0))
    y1 = int(round((h - th) / 2.0))
    return (img.crop(
        (x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))
