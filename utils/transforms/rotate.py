import random
import torchvision.transforms.functional as tf

from PIL import Image


class RandomRotate(object):

  def __init__(self, degree):
    self.degree = degree

  def __call__(self, img, mask):
    rotate_degree = random.random() * 2 * self.degree - self.degree
    return (
        tf.affine(
            img,
            translate=(0, 0),
            scale=1.0,
            angle=rotate_degree,
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0),
            shear=0.0,
        ),
        tf.affine(
            mask,
            translate=(0, 0),
            scale=1.0,
            angle=rotate_degree,
            resample=Image.NEAREST,
            fillcolor=250,
            shear=0.0,
        ),
    )


class RotateDegree(object):

  def __init__(self, degree):
    self.degree = degree

  def __call__(self, img, mask):
    return (
        tf.affine(
            img,
            translate=(0, 0),
            scale=1.0,
            angle=self.degree,
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0),
            shear=0.0,
        ),
        tf.affine(
            mask,
            translate=(0, 0),
            scale=1.0,
            angle=self.degree,
            resample=Image.NEAREST,
            fillcolor=250,
            shear=0.0,
        ),
    )
