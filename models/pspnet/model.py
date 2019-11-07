import torch.nn as nn
import torch.nn.functional as F

from .ppm import PPM

from ..base import BaseModel
from ..encoders import get_encoder


class PSPNet(BaseModel):

  def __init__(self,
               encoder_name='resnet50',
               encoder_weights='imagenet',
               bins=(1, 2, 3, 6),
               classes=2,
               BatchNorm=nn.BatchNorm2d,
               auxiliary_loss=False,
               auxloss_weight=0,
               criterion=None):
    super().__init__()
    self.criterion = criterion

    feature_dim = 2048
    self.ppm = PPM(feature_dim, int(feature_dim / len(bins)), bins, BatchNorm)

    self.decoder = nn.Sequential(
        nn.Conv2d(feature_dim * 2, 512, kernel_size=3, padding=1, bias=False),
        BatchNorm(512), nn.ReLU(inplace=True),
        nn.Conv2d(512, classes, kernel_size=1))

    self.auxiliary_loss = auxiliary_loss
    self.auxloss_weight = auxloss_weight
    # Add auxiliary layer for encoder.
    self.aux = nn.Sequential(
        nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
        BatchNorm(256), nn.ReLU(inplace=True),
        nn.Conv2d(256, classes, kernel_size=1))

    if encoder_weights == None:
      self.encoder = get_encoder(encoder_name, encoder_weights=encoder_weights)
      self.initialize()
    else:
      self.initialize()
      self.encoder = get_encoder(encoder_name, encoder_weights=encoder_weights)

  def forward(self, x, y=None):
    """ After encoder defined in a structure like ResNet. we'll get a feature
    map 8 times smaller than origianl image.
    """
    x_size = x.size()
    h, w = x_size[2:]
    if (x_size[2] - 1) % 8 == 0:
      align_corners = True
    elif x_size[2] % 8 == 0:
      align_corners = False
    else:
      raise 'Feature map size mismatch. Please check your encoder.'

    x = self.encoder(x)
    if self.auxiliary_loss:
      x_aux = x[1]
    x = x[0]

    x = self.ppm(x)
    x = self.decoder(x)
    x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=align_corners)

    if y is not None and self.criterion is not None:
      if self.auxiliary_loss:
        x_aux = self.aux(x_aux)
        x_aux = F.interpolate(x_aux, size=(h, w), mode='bilinear', align_corners=align_corners)

        main_loss = self.criterion(x, y)
        aux_loss = self.criterion(x_aux, y)
        return main_loss + aux_loss * self.auxloss_weight
      else:
        main_loss = self.criterion(x, y)
        return main_loss
    else:
      return x
