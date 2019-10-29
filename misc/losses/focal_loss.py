import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

  def __init__(self, gamma=2):
    super().__init__()
    self.gamma = gamma

  def forward(self, input, target):
    # Inspired by the implementation of binary_cross_entropy_with_logits
    if not (target.size() == input.size()):
      raise ValueError(
          "Target size ({}) must be the same as input size ({})".format(
              target.size(), input.size()))

    input = torch.sigmoid(input)

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() +
                                               (-input - max_val).exp()).log()

    # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
    invprobs = F.logsigmoid(-input * (target * 2 - 1))
    loss = (invprobs * self.gamma).exp() * loss

    return loss.mean()
