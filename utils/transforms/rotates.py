import cv2
import numpy as np


def rotate_90(image, label, prob):
  if prob:
    k = np.random.choice([-1, 0, 1], size=(), p=[prob / 2, 1 - prob, prob / 2])
    if k:
      image = np.rot90(image, k=k, axes=(0, 1))
      label = np.rot90(label, k=k, axes=(0, 1))
  return image, label


def rotate_angle(image, label, angle_max):
  if angle_max:
    # Random angle in range [-angle_max, angle_max]
    angle = np.random.choice(
        np.linspace(-angle_max, angle_max, num=21), size=())

    # Get parameters for affine transform
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform transform
    image = cv2.warpAffine(image, M, (nW, nH))
    label = cv2.warpAffine(label, M, (nW, nH))
  return image, label
