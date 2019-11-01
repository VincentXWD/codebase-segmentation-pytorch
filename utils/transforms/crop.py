import collections
import cv2
import random
import numbers


class RandomCrop(object):
  """Crops the given ndarray image (H*W*C or H*W).
  Args:
    size (sequence or int): Desired output size of the crop. If size is an
    int instead of sequence like (h, w), a square crop (size, size) is made.
  """

  def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
    if isinstance(size, int):
      self.crop_h = size
      self.crop_w = size
    elif isinstance(size, collections.Iterable) and len(size) == 2 \
        and isinstance(size[0], int) and isinstance(size[1], int) \
        and size[0] > 0 and size[1] > 0:
      self.crop_h = size[0]
      self.crop_w = size[1]
    else:
      raise (RuntimeError("crop size error.\n"))
    if crop_type == 'center' or crop_type == 'rand':
      self.crop_type = crop_type
    else:
      raise (RuntimeError("crop type error: rand | center\n"))
    if padding is None:
      self.padding = padding
    elif isinstance(padding, list):
      if all(isinstance(i, numbers.Number) for i in padding):
        self.padding = padding
      else:
        raise (RuntimeError("padding in Crop() should be a number list\n"))
      if len(padding) != 3:
        raise (RuntimeError("padding channel is not equal with 3\n"))
    else:
      raise (RuntimeError("padding in Crop() should be a number list\n"))
    if isinstance(ignore_label, int):
      self.ignore_label = ignore_label
    else:
      raise (RuntimeError("ignore_label should be an integer number\n"))

  def __call__(self, image, label):
    h, w = label.shape
    pad_h = max(self.crop_h - h, 0)
    pad_w = max(self.crop_w - w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
      if self.padding is None:
        raise (RuntimeError(
            "Crop() need padding while padding argument is None\n")
              )
      image = cv2.copyMakeBorder(
          image,
          pad_h_half,
          pad_h - pad_h_half,
          pad_w_half,
          pad_w - pad_w_half,
          cv2.BORDER_CONSTANT,
          value=self.padding)
      label = cv2.copyMakeBorder(
          label,
          pad_h_half,
          pad_h - pad_h_half,
          pad_w_half,
          pad_w - pad_w_half,
          cv2.BORDER_CONSTANT,
          value=self.ignore_label)
    h, w = label.shape
    if self.crop_type == 'rand':
      h_off = random.randint(0, h - self.crop_h)
      w_off = random.randint(0, w - self.crop_w)
    else:
      h_off = int((h - self.crop_h) / 2)
      w_off = int((w - self.crop_w) / 2)
    image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
    label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
    return image, label
