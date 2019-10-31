import torch

import network
from misc.sync_batchnorm.batchnorm import convert_model

from utils.transforms import Compose, Scale, RandomRotate, RandomHorizontallyFlip


def train_epoch():
  pass


def train():
  pass


def main():
  augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])
  model = network.get_model()
  # TODO(xwd): add flag [convert_sync_bn]
  model = convert_model(model)


if __name__ == '__main__':
  main()
