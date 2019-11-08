import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):

  def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
    """ PPM multi-scale module for PSPNet. Please note that if you have image size that
    can be divided by 8, we will not use align_corner for bilinear interpolate since it
    may let edge leak.
    """
    super(PPM, self).__init__()
    self.features = []
    for bin in bins:
      self.features.append(
          nn.Sequential(
              nn.AdaptiveAvgPool2d(bin),
              nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
              BatchNorm(reduction_dim),
              nn.ReLU(inplace=True)))
    self.features = nn.ModuleList(self.features)

  def forward(self, x):
    x_size = x.size()
    out = [x]

    if x_size[2] % 8 == 0:
      align_corners = False
    else:
      align_corners = True

    for f in self.features:
      out.append(
          F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=align_corners))

    return torch.cat(out, 1)
