import torch
import torch.nn as nn
from .basemodel import BaseModel


class EncoderDecoder(BaseModel):

  def __init__(self, encoder, decoder, activation=None):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

    if callable(activation):
      self.activation = activation
    elif activation == 'softmax':
      self.activation = nn.Softmax(dim=1)
    elif activation == 'sigmoid':
      self.activation = nn.Sigmoid()
    else:
      # Default: return the raw outputs.
      self.activation = lambda x: x

  def forward(self, x):
    """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
    x = self.encoder(x)
    y = self.decoder(x)
    return y

  def predict(self, x):
    """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
    if self.training:
      self.eval()

    with torch.no_grad():
      x = self.forward(x)
      if self.activation:
        x = self.activation(x)

    return x
