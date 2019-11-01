import numpy as np


class InterSectionAndUnion(object):

  def __init__(self, n_classes, ignore_index):
    self.n_classes = n_classes
    self.ignore_index = ignore_index

  def __call__(self, output, target):
    # n_classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to n_classes - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == self.ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(
        intersection, bins=np.arange(self.n_classes + 1))
    area_output, _ = np.histogram(output, bins=np.arange(self.n_classes + 1))
    area_target, _ = np.histogram(target, bins=np.arange(self.n_classes + 1))
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target
