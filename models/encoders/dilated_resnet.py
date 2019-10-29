"""Dilated ResNet"""
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import numpy as np

from pretrainedmodels.models.torchvision_models import pretrained_settings

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'BasicBlock', 'Bottleneck'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  "3x3 convolution with padding"

  kernel_size = np.asarray((3, 3))

  # Compute the size of the upsampled filter with
  # a specified dilation rate.
  upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

  # Determine the padding that is necessary for full padding,
  # meaning the output spatial size is equal to input spatial size
  full_padding = (upsampled_kernel_size - 1) // 2

  # Conv2d doesn't accept numpy arrays as arguments
  full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

  return nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=kernel_size,
      stride=stride,
      padding=full_padding,
      dilation=dilation,
      bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu1 = nn.ReLU(inplace=True)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes, dilation=dilation)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu2(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)

    #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
    #                       padding=1, bias=False)

    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self,
               block,
               layers,
               num_classes=1000,
               fully_conv=False,
               remove_avg_pool_layer=False,
               output_stride=32):

    # Add additional variables to track
    # output stride. Necessary to achieve
    # specified output stride.
    self.output_stride = output_stride
    self.current_stride = 4
    self.current_dilation = 1

    self.remove_avg_pool_layer = remove_avg_pool_layer

    self.inplanes = 64
    self.fully_conv = fully_conv
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(
        3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    self.avgpool = nn.AvgPool2d(7)

    self.fc = nn.Linear(512 * block.expansion, num_classes)

    if self.fully_conv:
      self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
      # In the latest unstable torch 4.0 the tensor.copy_
      # method was changed and doesn't work as it used to be
      #self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None

    if stride != 1 or self.inplanes != planes * block.expansion:

      # Check if we already achieved desired output stride.
      if self.current_stride == self.output_stride:

        # If so, replace subsampling with a dilation to preserve
        # current spatial resolution.
        self.current_dilation = self.current_dilation * stride
        stride = 1
      else:

        # If not, perform subsampling and update current
        # new output stride.
        self.current_stride = self.current_stride * stride

      # We don't dilate 1x1 convolution.
      downsample = nn.Sequential(
          nn.Conv2d(
              self.inplanes,
              planes * block.expansion,
              kernel_size=1,
              stride=stride,
              bias=False),
          nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride,
            downsample,
            dilation=self.current_dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
          block(self.inplanes, planes, dilation=self.current_dilation))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    if not self.remove_avg_pool_layer:
      x = self.avgpool(x)

    if not self.fully_conv:
      x = x.view(x.size(0), -1)

    x = self.fc(x)

    return x


class ResNetEncoder(ResNet):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.pretrained = False
    del self.fc

  def forward(self, x):
    x0 = self.conv1(x)
    x0 = self.bn1(x0)
    x0 = self.relu(x0)

    x1 = self.maxpool(x0)
    x1 = self.layer1(x1)

    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)

    return [x4, x3, x2, x1, x0]

  def load_state_dict(self, state_dict, **kwargs):
    state_dict.pop('fc.bias')
    state_dict.pop('fc.weight')
    super().load_state_dict(state_dict, **kwargs)


def resnet18(pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False, root='./pretrain_models', **kwargs):
  """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    from model_store import get_model_file
    model.load_state_dict(
        torch.load(get_model_file('resnet50', root=root)), strict=False)
  return model


def resnet101(pretrained=False, root='./pretrain_models', **kwargs):
  """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
  # Remove the following lines of comments
  # if u want to train from a pretrained model
  if pretrained:
    from model_store import get_model_file
    model.load_state_dict(
        torch.load(get_model_file('resnet101', root=root)), strict=False)
  return model


def resnet152(pretrained=False, root='~/.encoding/models', **kwargs):
  """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
  model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
  if pretrained:
    # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    model.load_state_dict(
        torch.load('./pretrain_models/resnet152-b121ed2d.pth'), strict=False)
  return model


dilated_resnet_encoders = {
    'dilated_resnet18': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet18'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },
    'dilated_resnet34': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet34'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },
    'dilated_resnet50': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet50'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },
    'dilated_resnet101': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet101'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
        },
    },
    'dilated_resnet152': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet152'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 8, 36, 3],
        },
    },
}
