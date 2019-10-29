import cv2
import numpy as np


def random_noise(image, std):
  # Gaussian noise
  if std:
    noise = np.random.normal(0, std, size=image.shape)
    image = image + noise
    image[image < 0] = 0
    image[image > 255] = 255
    image = image.astype(np.uint8)
  return image


def resize_image(image,
                 expected_size,
                 pad_value,
                 ret_params=False,
                 mode=cv2.INTER_LINEAR):
  """
  image (ndarray) with either shape of [H,W,3] for RGB or [H,W] for grayscale.
  Padding is added so that the content of image is in the center.
  """
  h, w = image.shape[:2]
  if w > h:
    w_new = int(expected_size)
    h_new = int(h * w_new / w)
    image = cv2.resize(image, (w_new, h_new), interpolation=mode)

    pad_up = (w_new - h_new) // 2
    pad_down = w_new - h_new - pad_up
    if len(image.shape) == 3:
      pad_width = ((pad_up, pad_down), (0, 0), (0, 0))
      constant_values = ((pad_value, pad_value), (0, 0), (0, 0))
    elif len(image.shape) == 2:
      pad_width = ((pad_up, pad_down), (0, 0))
      constant_values = ((pad_value, pad_value), (0, 0))

    image = np.pad(
        image,
        pad_width=pad_width,
        mode="constant",
        constant_values=constant_values,
    )
    if ret_params:
      return image, pad_up, 0, h_new, w_new
    else:
      return image

  elif w < h:
    h_new = int(expected_size)
    w_new = int(w * h_new / h)
    image = cv2.resize(image, (w_new, h_new), interpolation=mode)

    pad_left = (h_new - w_new) // 2
    pad_right = h_new - w_new - pad_left
    if len(image.shape) == 3:
      pad_width = ((0, 0), (pad_left, pad_right), (0, 0))
      constant_values = ((0, 0), (pad_value, pad_value), (0, 0))
    elif len(image.shape) == 2:
      pad_width = ((0, 0), (pad_left, pad_right))
      constant_values = ((0, 0), (pad_value, pad_value))

    image = np.pad(
        image,
        pad_width=pad_width,
        mode="constant",
        constant_values=constant_values,
    )
    if ret_params:
      return image, 0, pad_left, h_new, w_new
    else:
      return image

  else:
    image = cv2.resize(
        image, (expected_size, expected_size), interpolation=mode)
    if ret_params:
      return image, 0, 0, expected_size, expected_size
    else:
      return image
