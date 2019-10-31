import random
import torchvision.transforms.functional as tf


class AdjustGamma(object):

  def __init__(self, gamma):
    self.gamma = gamma

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):

  def __init__(self, saturation):
    self.saturation = saturation

  def __call__(self, img, mask):
    assert img.size == mask.size
    return (
        tf.adjust_saturation(
            img, random.uniform(1 - self.saturation, 1 + self.saturation)),
        mask,
    )


class AdjustHue(object):

  def __init__(self, hue):
    self.hue = hue

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_hue(img, random.uniform(-self.hue, self.hue)), mask


class AdjustBrightness(object):

  def __init__(self, bf):
    self.bf = bf

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_brightness(img, random.uniform(1 - self.bf,
                                                    1 + self.bf)), mask


class AdjustContrast(object):

  def __init__(self, cf):
    self.cf = cf

  def __call__(self, img, mask):
    assert img.size == mask.size
    return tf.adjust_contrast(img, random.uniform(1 - self.cf,
                                                  1 + self.cf)), mask
