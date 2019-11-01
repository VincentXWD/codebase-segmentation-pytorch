import numpy as np


class DiceCoeff(object):
  """ Dice coefficient for binary segmentation evaluations.
  """

  def __init__(self, empty_score=1.0):
    self.empty_score = empty_score

  def __call__(self, output, target):
    output = np.asarray(output).astype(np.bool)
    target = np.asarray(target).astype(np.bool)

    if output.shape != target.shape:
      raise ValueError(
          "Shape mismatch: output and target must have the same shape.")

    output = output >= 0.5
    target = target >= 0.5

    im_sum = output.sum() + target.sum()
    if im_sum == 0:
      return self.empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(output, target)

    return 2. * intersection.sum() / im_sum
