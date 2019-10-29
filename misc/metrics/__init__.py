import numpy as np

from medpy import metric
from sklearn.metrics import roc_curve, auc


def dice_coeff(im1, im2, empty_score=1.0):
  """Calculates the dice coefficient for the images"""

  im1 = np.asarray(im1).astype(np.bool)
  im2 = np.asarray(im2).astype(np.bool)

  if im1.shape != im2.shape:
    raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

  im1 = im1 >= 0.5
  im2 = im2 >= 0.5

  im_sum = im1.sum() + im2.sum()
  if im_sum == 0:
    return empty_score

  # Compute Dice coefficient
  intersection = np.logical_and(im1, im2)

  return 2. * intersection.sum() / im_sum


def get_auc(pred, label):
  pred = pred.ravel()
  label = label.ravel()

  fpr, tpr, thresholds = roc_curve(label, pred, pos_label=1)
  return auc(fpr, tpr)


def get_scores(pred, label):
  volscores = {}
  volscores['dice'] = dice_coeff(pred, label)
  volscores['auc'] = get_auc(pred, label)
  volscores['recall'] = metric.recall(pred, label)
  volscores['precision'] = metric.precision(pred, label)
  return volscores


if __name__ == '__main__':
  pass
