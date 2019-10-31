import random

from .crop import RandomCrop

from PIL import Image


class FreeScale(object):

  def __init__(self, size):
    self.size = tuple(reversed(size))  # size: (h, w)

  def __call__(self, img, mask):
    assert img.size == mask.size
    return (img.resize(self.size,
                       Image.BILINEAR), mask.resize(self.size, Image.NEAREST))


class Scale(object):

  def __init__(self, size):
    self.size = size

  def __call__(self, img, mask):
    assert img.size == mask.size
    w, h = img.size
    if (w >= h and w == self.size) or (h >= w and h == self.size):
      return img, mask
    if w > h:
      ow = self.size
      oh = int(self.size * h / w)
      return (img.resize((ow, oh),
                         Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))
    else:
      oh = self.size
      ow = int(self.size * w / h)
      return (img.resize((ow, oh),
                         Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST))


class RandomSized(object):

  def __init__(self, size):
    self.size = size
    self.scale = Scale(self.size)
    self.crop = RandomCrop(self.size)

  def __call__(self, img, mask):
    assert img.size == mask.size

    w = int(random.uniform(0.5, 2) * img.size[0])
    h = int(random.uniform(0.5, 2) * img.size[1])

    img, mask = (img.resize((w, h),
                            Image.BILINEAR), mask.resize((w, h), Image.NEAREST))

    return self.crop(*self.scale(img, mask))
