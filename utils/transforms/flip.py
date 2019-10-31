import random
from PIL import Image


class RandomHorizontallyFlip(object):

  def __init__(self, p):
    self.p = p

  def __call__(self, img, mask):
    if random.random() < self.p:
      return (img.transpose(Image.FLIP_LEFT_RIGHT),
              mask.transpose(Image.FLIP_LEFT_RIGHT))
    return img, mask


class RandomVerticallyFlip(object):

  def __init__(self, p):
    self.p = p

  def __call__(self, img, mask):
    if random.random() < self.p:
      return (img.transpose(Image.FLIP_TOP_BOTTOM),
              mask.transpose(Image.FLIP_TOP_BOTTOM))
    return img, mask
