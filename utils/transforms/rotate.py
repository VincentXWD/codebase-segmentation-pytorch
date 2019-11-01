import collections
import numbers
import random
import cv2


class RandomRotate(object):

  def __init__(self, rotate, padding, ignore_label=255, p=0.5):
    assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
    if isinstance(rotate[0], numbers.Number) and isinstance(
        rotate[1], numbers.Number) and rotate[0] < rotate[1]:
      self.rotate = rotate
    else:
      raise (RuntimeError("RandRotate() scale param error.\n"))
    assert padding is not None
    assert isinstance(padding, list) and len(padding) == 3
    if all(isinstance(i, numbers.Number) for i in padding):
      self.padding = padding
    else:
      raise (RuntimeError("padding in RandRotate() should be a number list\n"))
    assert isinstance(ignore_label, int)
    self.ignore_label = ignore_label
    self.p = p

  def __call__(self, image, label):
    if random.random() < self.p:
      angle = self.rotate[0] + (self.rotate[1] -
                                self.rotate[0]) * random.random()
      h, w = label.shape
      matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
      image = cv2.warpAffine(
          image,
          matrix, (w, h),
          flags=cv2.INTER_LINEAR,
          borderMode=cv2.BORDER_CONSTANT,
          borderValue=self.padding)
      label = cv2.warpAffine(
          label,
          matrix, (w, h),
          flags=cv2.INTER_NEAREST,
          borderMode=cv2.BORDER_CONSTANT,
          borderValue=self.ignore_label)
    return image, label
