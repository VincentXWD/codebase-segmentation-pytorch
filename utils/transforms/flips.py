import numpy as np


def flip_horizon(image, label, prob):
  if prob:
    if np.random.choice([False, True], size=(), p=[1 - prob, prob]):
      image = np.flip(image, axis=1)
      label = np.flip(label, axis=1)
  return image, label
