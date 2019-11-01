import functools
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders
from .dpn import dpn_encoders
from .vgg import vgg_encoders
from .senet import senet_encoders
from .densenet import densenet_encoders
from .inceptionresnetv2 import inception_encoders
from .dilated_resnet import dilated_resnet_encoders
from .octave_resnet import octave_resnet_encoders

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inception_encoders)
encoders.update(octave_resnet_encoders)
encoders.update(dilated_resnet_encoders)


def get_encoder(name, encoder_weights=None):
  Encoder = encoders[name]['encoder']
  encoder = Encoder(**encoders[name]['params'])
  encoder.out_shapes = encoders[name]['out_shapes']

  if encoder_weights is not None:
    settings = encoders[name]['pretrained_settings'][encoder_weights]
    encoder.load_state_dict(model_zoo.load_url(settings['url']))

  return encoder


def get_encoder_names():
  return list(encoders.keys())
