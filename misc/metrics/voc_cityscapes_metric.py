import numpy as np


class VocCityscapesMetric(object):

  def __init__(self):
    np.seterr(divide='ignore', invalid='ignore')

    # TODO(xwd): Change it if use other number of classes.
    self.names = [
        'unlabeled', 'ego', 'rectification', 'out', 'static', 'dynamic',
        'ground', 'road', 'sidewalk', 'parking', 'rail', 'building', 'wall',
        'fence', 'guard', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic',
        'traffic', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
        'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle',
        'license'
    ]
    self.iu = None
    self.mean_pixel_acc = None
    self.mean_IU = None
    self.calculated = False

  def __call__(self, all_results, classes):
    hist = np.zeros((classes, classes))
    correct = 0
    labeled = 0
    count = 0
    for d in all_results:
      hist += d['hist']
      correct += d['correct']
      labeled += d['labeled']
      count += 1

    self.iu, self.mean_IU, self.mean_pixel_acc = self._compute_score(
        hist, correct, labeled)
    self.calculated = True

  def __repr__(self):
    n = self.iu.size
    lines = list()
    lines.append('\n')
    for i in range(n):
      if self.names is None:
        _cls = f'Class {i + 1}:'
      else:
        _cls = f'{i + 1} {self.names[i]}'

      lines.append(f'{_cls}\t{self.iu[i] * 100}%')
    lines.append('\n')
    lines.append(
        f'mean_IU\t{self.mean_IU * 100}%\tmean_pixel_ACC\t{self.mean_pixel_acc * 100}%'
    )
    line = '\n'.join(lines)

    return line

  def _compute_score(self, hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_pixel_acc

  def hist_info(self, n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    # just evaluation the pixel assigned value 0 <= trainID < n_cl,
    # keep uneval pixel assigned -1 or 255 away.
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))
    return np.bincount(
        n_cl * gt[k].astype(int) + pred[k].astype(int),
        minlength=n_cl**2).reshape(n_cl, n_cl), labeled, correct
