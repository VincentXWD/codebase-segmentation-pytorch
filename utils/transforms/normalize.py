import numpy as np


def normalize_per_channel(img):
  img[:, :, 0] = (img[:, :, 0] - np.mean(img[:, :, 0])) / np.std(img[:, :, 0])
  img[:, :, 1] = (img[:, :, 1] - np.mean(img[:, :, 1])) / np.std(img[:, :, 1])
  img[:, :, 2] = (img[:, :, 2] - np.mean(img[:, :, 2])) / np.std(img[:, :, 2])
  return img


def normalize_per_image(img,
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]):
  mean = np.array(mean)[None, None, :]
  std = np.array(std)[None, None, :]

  img = img.astype(np.float32) / 255.0
  img = (img - mean) / std

  return img
