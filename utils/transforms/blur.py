import cv2
import random


class RandomGaussianBlur(object):

  def __init__(self, radius=5):
    self.radius = radius

  def __call__(self, image, label):
    if random.random() < 0.5:
      image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
    return image, label
