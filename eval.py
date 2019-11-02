"""
@author: Wendong Xu
@contact: kirai.wendong@gmail.com
@file: train.py
@time: 2019-11-02 20:11
@desc:
"""
# Use CUDA by default. Currently can only use one GPU. For this GPU, calculate only a batch with size 1 at a time.
# It's easy to control the data streaming when use multi-GPUs even multi-workers.
# # TODO(xwd): Implement concurrent & parallel by myself.
import argparse
import os
import torch
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

import network

from torch.utils.data import DataLoader

from tools.statistics import count_model_param
from utils.safe_loader import safe_loader
from utils.file_ops import check_dir_exists
from misc.metrics.average_meter import AverageMeter
from misc.metrics.voc_cityscapes_metric import VocCityscapesMetric
from misc import get_logger, config
from dataset import cityscapes


def get_logger_and_parser():
  parser = argparse.ArgumentParser(description='config')
  parser.add_argument(
      '--config',
      type=str,
      default='config/cityscapes.yaml',
      help='Configuration file to use',
  )

  args = parser.parse_args()

  assert args.config is not None
  cfg = config.load_cfg_from_cfg_file(args.config)
  args_dict = dict()
  for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
  args_dict.update(cfg)

  run_dir = os.path.join('runs',
                         os.path.basename(args.config)[:-5], cfg['exp_name'])
  check_dir_exists(run_dir)

  run_id = str(int(time.time()))
  logger = get_logger(run_dir, run_id, 'val')

  logger.info('RUNDIR: {}'.format(run_dir))

  return logger, cfg, run_dir


def get_dataloader(cfg):
  # TODO(xwd): Adaptive normalization by some large image.
  # E.g. In medical image processing, WSI image is very large and different to ordinary images.
  val_data = cityscapes.Cityscapes(
      cfg['data_path'], split='val', transform=None)

  val_loader = DataLoader(
      val_data,
      batch_size=1,
      shuffle=False,
      num_workers=8,
      pin_memory=True,
      drop_last=False)

  return val_loader


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
    output = F.interpolate(
        output, (h_i, w_i), mode='bilinear', align_corners=True)
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


def eval(cfg, logger, model, eval_loader):
  value_scale = 255
  mean = [0.485, 0.456, 0.406]
  mean = [item * value_scale for item in mean]
  std = [0.229, 0.224, 0.225]
  std = [item * value_scale for item in std]

  ruler = VocCityscapesMetric()
  eval_results = []
  data_time = AverageMeter()
  batch_time = AverageMeter()
  end = time.time()
  model.eval()

  for i, (image, label) in enumerate(eval_loader):
    data_time.update(time.time() - end)
    image = np.squeeze(image.numpy(), axis=0)
    label = np.squeeze(label.numpy(), axis=0)

    h, w, _ = image.shape
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
          image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
      prediction += eval_in_scale(model, image_scale, cfg['classes'],
                                  cfg['test_h'], cfg['test_w'], h, w, mean, std)

    prediction /= len(cfg['scales'])
    # N(WHk)
    prediction = np.argmax(prediction, axis=2)
    batch_time.update(time.time() - end)
    end = time.time()

    if ((i + 1) % 10 == 0) or (i + 1 == len(eval_loader)):
      logger.info('Test: [{}/{}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(
                      i + 1,
                      len(eval_loader),
                      data_time=data_time,
                      batch_time=batch_time))

    if label is not None:
      hist_tmp, labeled_tmp, correct_tmp = ruler.hist_info(
          cfg['classes'], prediction, label)
      eval_results.append({
          'hist': hist_tmp,
          'labeled': labeled_tmp,
          'correct': correct_tmp
      })

  ruler(eval_results, cfg['classes'])

  if ruler.calculated:
    logger.info(ruler)


def main():
  logger, cfg, run_dir = get_logger_and_parser()

  model_path = os.path.join(run_dir, 'model')
  model = network.get_model()
  if cfg['multi_gpu']:
    model = nn.DataParallel(model).cuda()
  logger.info(
      f'Segmentation Network Total Params number: {count_model_param(model) / 1E6}M'
  )
  check_dir_exists(model_path)

  # Get dataloader.
  eval_loader = get_dataloader(cfg)

  # Load model.
  checkpoint = torch.load(cfg['model_path'])
  model.load_state_dict(
      safe_loader(
          checkpoint['state_dict'],
          use_model='multi' if cfg['multi_gpu'] else 'single'))

  eval(cfg, logger, model, eval_loader)


if __name__ == '__main__':
  main()
