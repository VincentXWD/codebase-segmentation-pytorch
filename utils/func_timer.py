import time


def func_timer(function):

  @wraps(function)
  def function_timer(*args, **kwargs):
    print('[Function: {name} start...]'. \
                format(name=function.__name__))
    t0 = time.time()
    result = function(*args, **kwargs)
    t1 = time.time()
    print('[Function: {name} finished, spent time: {time:.2f}s]'. \
                format(name=function.__name__, time=t1 - t0))
    return result

  return function_timer