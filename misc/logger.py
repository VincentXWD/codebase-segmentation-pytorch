import os
import logging


def get_logger(log_path, run_id, suffix):
  logger = logging.getLogger(suffix)
  logger.setLevel(logging.INFO)
  fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

  ch = logging.StreamHandler()
  ch.setFormatter(logging.Formatter(fmt))
  logger.addHandler(ch)

  log_name = os.path.join(log_path, run_id + '_' + suffix + '.log')
  fh = logging.FileHandler(log_name, 'a', encoding='utf-8')
  fh.setFormatter(logging.Formatter(fmt))
  fh.setLevel(logging.INFO)
  logger.addHandler(fh)

  return logger


if __name__ == '__main__':
  pass
