import torch
import torch.nn as nn


def dice_loss(input, target):
  input = torch.sigmoid(input)
  smooth = 1e-5

  iflat = input.view(-1)
  tflat = target.view(-1)
  intersection = (iflat * tflat).sum()

  return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


def focal_loss(input,
               target,
               reduction='mean',
               beta=0.5,
               gamma=2.,
               eps=1e-7,
               **kwargs):
  """
  Focal loss, see arXiv:1708.02002
  input: [B, 1, H, W] tensor that contains predictions to compare
  target: [B, 1, H, W] tensor that contains targets to compare to
  reduction: one of mean, sum or none. Used to choose how loss is reduced
         over batches
  beta: weight in [0; 1] to give to positive targets. The higher it is, the
      more true positive and false negative are important. Negative targets
      have weight 1-beta
  gamma: parameter that reduces the loss contribution from easy examples and
       extends the range in which an example receives low loss. It also
       gives more weight to misclassified examples
  eps: constant used for numerical stability
  return: [1] or [B] (if reduction='none') tensor containing loss between
      input and target
  """
  n = input.size(0)
  iflat = torch.sigmoid(input).view(n, -1).clamp(eps, 1 - eps)
  tflat = target.view(n, -1)
  focal = -(beta * tflat * (1 - iflat).pow(gamma) * iflat.log() + (1 - beta) *
            (1 - tflat) * iflat.pow(gamma) * (1 - iflat).log()).mean(-1)
  if reduction == 'mean':
    return focal.mean()
  elif reduction == 'sum':
    return focal.sum()
  else:
    return focal


class FocalDiceLoss(nn.Module):
  """
  Weighted linear combination of focal and dice losses
  a: weight of binary cross-entropy
  b: weight of dice
  smooth: value added to both numerator and denominator of dice to avoid
      division by zero and smooth gradient around 0
  beta: weight in [0; 1] to give to positive targets. The higher it is,
      the more true positive and false negative are important. Negative
      targets have weight 1-beta
  gamma: parameter that reduces the loss contribution from easy examples
       and extends the range in which an example receives low loss. It
       also gives more weight to misclassified examples
  reduction: one of mean, sum or none. Used to choose how loss is reduced
         over batches
  """

  def __init__(self,
               a=0.5,
               b=0.5,
               smooth=1.,
               beta=0.5,
               gamma=2.,
               reduction='mean'):
    super().__init__()
    self.a = a
    self.b = b
    self.smooth = smooth
    self.beta = beta
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, input, target):
    """
    input: [B, 1, H, W] tensor that contains predictions to compare
    target: [B, 1, H, W] tensor that contains targets to compare to
    return: [1] or [B] (if self.reduction='none') tensor containing loss
        between input and target
    """
    focal = focal_loss(
        input,
        target,
        beta=self.beta,
        gamma=self.gamma,
        reduction=self.reduction)
    dice = dice_loss(input, target)
    return self.a * focal + self.b * dice
