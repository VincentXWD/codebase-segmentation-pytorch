import numpy as np


def calc_img_mean_and_std(img):
  m0 = np.mean(img[:, :, 0]) / 255.0
  s0 = np.std(img[:, :, 0]) / 255.0
  m1 = np.mean(img[:, :, 1]) / 255.0
  s1 = np.std(img[:, :, 1]) / 255.0
  m2 = np.mean(img[:, :, 2]) / 255.0
  s2 = np.std(img[:, :, 2]) / 255.0
  return [[m0, m1, m2], [s0, s1, s2]]