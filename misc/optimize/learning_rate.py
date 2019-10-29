def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
  """Sets the learning rate to the base LR decayed by 10 every step epochs"""
  lr = base_lr * (multiplier**(epoch // step_epoch))
  return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
  """poly learning rate policy"""
  lr = base_lr * (1 - float(curr_iter) / max_iter)**power
  return lr
