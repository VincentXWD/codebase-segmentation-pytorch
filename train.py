import torch

import network

from utils.transforms import Compose, Scale, RandomRotate, RandomHorizontallyFlip


if __name__ == '__main__':
  augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])
  model = network.get_model()