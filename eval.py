"""
@author: Wendong Xu
@contact: kirai.wendong@gmail.com
@file: eval.py
@time: 2019-11-02 20:11
@desc:
"""
# Use CUDA by default.
import argparse
import os
import torch
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torch.multiprocessing as mp

import network

from torch.utils.data import DataLoader, Subset

from tools.statistics import count_model_param
from utils.safe_loader import safe_loader
from utils.file_ops import check_dir_exists
from misc.metrics.average_meter import AverageMeter
from misc.metrics.voc_cityscapes_metric import VocCityscapesMetric
from misc import get_logger, config
from dataset import cityscapes


def get_logger_and_parser():
  parser = argparse.ArgumentParser(description='config')
  parser.add_argument('--config', type=str, default='config/cityscapes_pspnet.yaml', help='Configuration file to use')
  parser.add_argument('--num_of_gpus', type=int, default=0)
  parser.add_argument('opts', help='', default=None, nargs=argparse.REMAINDER)

  args = parser.parse_args()

  assert args.config is not None
  cfg = config.load_cfg_from_cfg_file(args.config)

  if args.opts is not None:
      cfg = config.merge_cfg_from_list(cfg, args.opts)
  args_dict = dict()

  for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
  cfg.update(args_dict)


  run_dir = os.path.join('runs',
                         os.path.basename(args.config)[:-5], cfg['exp_name'])
  check_dir_exists(run_dir)

  run_id = str(int(time.time()))
  logger = get_logger(run_dir, run_id, 'val')

  logger.info('RUNDIR: {}'.format(run_dir))

  return logger, cfg, run_dir


def get_dataset(cfg):
  # TODO(xwd): Adaptive normalization by some large image.
  # E.g. In medical image processing, WSI image is very large and different to ordinary images.
  eval_dataset = cityscapes.Cityscapes(
      cfg['data_path'], split='val', transform=None)

  return eval_dataset


def eval_each(model, image, mean, std=None, flip=True):
  image = torch.from_numpy(image.transpose((2, 0, 1))).float()
  if std is None:
    for t, m in zip(image, mean):
      t.sub_(m)
  else:
    for t, m, s in zip(image, mean, std):
      t.sub_(m).div_(s)
  image = image.unsqueeze(0).cuda()
  if flip:
    image = torch.cat([image, image.flip(3)], 0)

  with torch.no_grad():
    output = model(image)

  _, _, h_i, w_i = image.shape
  _, _, h_o, w_o = output.shape
  if (h_o != h_i) or (w_o != w_i):
    if h_o % 8 == 0:
      align_corners = False
    elif (h_o - 1) % 8 == 0:
      align_corners = True
    output = F.interpolate(
        output, (h_i, w_i), mode='bilinear', align_corners=align_corners)
  output = F.softmax(output, dim=1)

  if flip:
    output = (output[0] + output[1].flip(2)) / 2
  else:
    output = output[0]

  output = output.data.cpu().numpy()
  output = output.transpose(1, 2, 0)
  return output


def eval_in_scale(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2 / 3):
  ori_h, ori_w, _ = image.shape
  pad_h = max(crop_h - ori_h, 0)
  pad_w = max(crop_w - ori_w, 0)
  pad_h_half = int(pad_h / 2)
  pad_w_half = int(pad_w / 2)
  if pad_h > 0 or pad_w > 0:
    image = cv2.copyMakeBorder(
        image,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=mean)

  new_h, new_w, _ = image.shape
  stride_h = int(np.ceil(crop_h * stride_rate))
  stride_w = int(np.ceil(crop_w * stride_rate))
  grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
  grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
  prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
  count_crop = np.zeros((new_h, new_w), dtype=float)

  for index_h in range(0, grid_h):
    for index_w in range(0, grid_w):
      s_h = index_h * stride_h
      e_h = min(s_h + crop_h, new_h)
      s_h = e_h - crop_h
      s_w = index_w * stride_w
      e_w = min(s_w + crop_w, new_w)
      s_w = e_w - crop_w
      image_crop = image[s_h:e_h, s_w:e_w].copy()
      count_crop[s_h:e_h, s_w:e_w] += 1
      prediction_crop[s_h:e_h, s_w:e_w, :] += eval_each(model, image_crop, mean,
                                                        std)

  prediction_crop /= np.expand_dims(count_crop, 2)
  prediction_crop = prediction_crop[pad_h_half:pad_h_half +
                                    ori_h, pad_w_half:pad_w_half + ori_w]
  prediction = cv2.resize(
      prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
  return prediction


def eval_single_gpu(worker, cfg, logger, eval_dataset, results_queue):
  os.environ["CUDA_VISIBLE_DEVICES"] = str(worker)

  value_scale = 255
  mean = [0.485, 0.456, 0.406]
  mean = [item * value_scale for item in mean]
  std = [0.229, 0.224, 0.225]
  std = [item * value_scale for item in std]

  ruler = VocCityscapesMetric()
  data_time = AverageMeter()
  batch_time = AverageMeter()
  end = time.time()

  # Load dataset
  data_size = len(eval_dataset)
  eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

  # Get model checkpoint.
  checkpoint = torch.load(cfg['model_path'])

  # Load model.
  model = network.get_model().cuda()
  model.load_state_dict(
      safe_loader(
          checkpoint['state_dict'],
          use_model='single'))
  logger.info(
      f'Worker[{worker}]: Segmentation Network Total Params number: {count_model_param(model) / 1E6}M'
  )

  model.eval()
  process_count = 0

  for _, (image, label) in enumerate(eval_loader):
    assert image.shape[0] == label.shape[0]
    data_time.update(time.time() - end)
    for j in range(image.shape[0]):
      cur_image = image[j].numpy()
      cur_label = label[j].numpy()

      h, w, _ = cur_image.shape
      prediction = np.zeros((h, w, cfg['classes']), dtype=float)

      for scale in cfg['scales']:
        long_size = round(scale * cfg['base_size'])
        new_h = long_size
        new_w = long_size
        if h > w:
          new_w = round(long_size / float(h) * w)
        else:
          new_h = round(long_size / float(w) * h)
        image_scale = cv2.resize(
            cur_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        prediction += eval_in_scale(model, image_scale, cfg['classes'],
                                    cfg['test_h'], cfg['test_w'], h, w, mean,
                                    std)

      prediction /= len(cfg['scales'])
      prediction = np.argmax(prediction, axis=2)
      batch_time.update(time.time() - end)
      end = time.time()
      process_count += 1

      if process_count % 10 == 0 or process_count == data_size:
        logger.info('[Worker{}] Test: [{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(
                        worker,
                        process_count,
                        data_size,
                        data_time=data_time,
                        batch_time=batch_time))

      if cur_label is not None:
        hist_tmp, labeled_tmp, correct_tmp = ruler.hist_info(
            cfg['classes'], prediction, cur_label)
        results_queue.put({
            'hist': hist_tmp,
            'labeled': labeled_tmp,
            'correct': correct_tmp
        })


def main():
  logger, cfg, run_dir = get_logger_and_parser()
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1' if cfg['num_of_gpus'] <= 0 else ','.join(str(x) for x in range(cfg['num_of_gpus']))
  cfg['multi_gpu'] = True if cfg['num_of_gpus'] > 1 else False

  model_path = os.path.join(run_dir, 'model')
  check_dir_exists(model_path)

  # Get dataset.
  eval_dataset = get_dataset(cfg)
  eval_datasize = len(eval_dataset)

  results_queue = mp.Queue(eval_datasize)

  if cfg['num_of_gpus'] == 1:
    eval_single_gpu(0, cfg, logger, eval_dataset, results_queue)
  else:
    # Multi-GPUs processing.
    stride = int(np.ceil(eval_datasize / cfg['num_of_gpus']))
    dataset_idx = list(range(eval_datasize))
    procs_list = []
    for n in range(cfg['num_of_gpus']):
      e_record = min((n + 1) * stride, eval_datasize)
      idx = dataset_idx[n * stride:e_record]
      p = mp.Process(target=eval_single_gpu, args=(n, cfg, logger, Subset(eval_dataset, idx),
                                                   results_queue,))
      procs_list.append(p)
      p.start()

  if cfg['split'] != 'test':
    eval_results = []
    for _ in range(eval_datasize):
        t = results_queue.get()
        eval_results.append(t)

    logger.info('Results accumulated.')

    ruler = VocCityscapesMetric()
    ruler(eval_results, cfg['classes'])

    if ruler.calculated:
      logger.info(ruler)

  if cfg['num_of_gpus'] > 1:
    for p in procs_list:
      p.join()


if __name__ == '__main__':
  try:
    mp.set_start_method('spawn')
  except RuntimeError:
    pass

  main()
