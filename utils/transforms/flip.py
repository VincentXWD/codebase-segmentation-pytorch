import cv2
import random


class RandomHorizontallyFlip(object):

  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, image, label):
    if random.random() < self.p:
      image = cv2.flip(image, 1)
      label = cv2.flip(label, 1)
    return image, label


class RandomVerticallyFlip(object):

  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, image, label):
    if random.random() < self.p:
      image = cv2.flip(image, 0)
      label = cv2.flip(label, 0)
    return image, label