import numpy as np


def random_crop(image, label, crop_range):
  """
  cropped image is a square.
  image (ndarray) with shape [H,W,3]
  label (ndarray) with shape [H,W]
  crop_ratio (list) contains 2 bounds
  """
  ##### Exception #####
  if crop_range[0] == crop_range[1] and crop_range[0] == 1.0:
    return image, label

  # Get random crop_ratio
  crop_ratio = np.random.choice(
      np.linspace(crop_range[0], crop_range[1], num=10), size=())

  # Get random coordinates
  H, W = label.shape
  size = H if H < W else W
  size = int(size * crop_ratio)
  max_i, max_j = H - size, W - size
  i = np.random.choice(np.arange(0, max_i + 1), size=())
  j = np.random.choice(np.arange(0, max_j + 1), size=())

  # Crop
  image_cropped = image[i:i + size, j:j + size, :]
  label_cropped = label[i:i + size, j:j + size]

  return image_cropped, label_cropped
