import torch
import torch.nn as nn
from .basemodel import BaseModel


class EncoderDecoder(BaseModel):

  def __init__(self, encoder, decoder, criterion=None):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.criterion = criterion

  def forward(self, x, y=None):
    """Sequentially pass `x` trough model`s `encoder` and `decoder` (return LOGITS).
       Use self.criterion to calculate iif both criterion and y are not `None`."""
    x = self.encoder(x)
    x = self.decoder(x)

    if self.criterion is not None and y is not None:
      loss = self.criterion(x, y)
      return loss
    else:
      return x
