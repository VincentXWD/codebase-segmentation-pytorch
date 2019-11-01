"""
@author: Wendong Xu
@contact: kirai.wendong@gmail.com
@file: train.py
@time: 2019-11-01 19:17
@desc:
"""
# Use CUDA by default.
# TODO(xwd): Support tensorboardX.

import argparse
import shutil
import os
import torch
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import network
import utils.transforms as transform

from torch.utils.data import DataLoader

from tools.statistics import count_model_param
from utils.checkpoint import read_newest_model_path
from utils.safe_loader import safe_loader
from utils.file_ops import check_dir_exists
from misc.metrics.average_meter import AverageMeter
from misc.metrics.intersection_union import InterSectionAndUnion
from misc.sync_batchnorm.batchnorm import convert_model
from misc import get_logger, config
from misc.optimize import poly_learning_rate
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
  logger = get_logger(run_dir, run_id, 'train')

  logger.info('RUNDIR: {}'.format(run_dir))

  shutil.copy(args.config, run_dir)

  return logger, cfg, run_dir


def get_dataloader(cfg):
  # TODO(xwd): Adaptive normalization by some large image.
  # E.g. In medical image processing, WSI image is very large and different to ordinary images.

  value_scale = 255
  mean = [0.485, 0.456, 0.406]
  mean = [item * value_scale for item in mean]
  std = [0.229, 0.224, 0.225]
  std = [item * value_scale for item in std]

  train_transform = transform.Compose([
      transform.RandomScale([cfg['scale_min'], cfg['scale_max']]),
      transform.RandomRotate([cfg['rotate_min'], cfg['rotate_max']],
                             padding=mean,
                             ignore_label=cfg['ignore_label']),
      transform.RandomGaussianBlur(),
      transform.RandomHorizontallyFlip(),
      transform.RandomCrop([cfg['train_h'], cfg['train_w']],
                           crop_type='rand',
                           padding=mean,
                           ignore_label=cfg['ignore_label']),
      transform.ToTensor(),
      transform.Normalize(mean=mean, std=std)
  ])

  train_data = cityscapes.Cityscapes(
      cfg['data_path'], split='train', transform=train_transform)

  train_loader = DataLoader(
      train_data,
      batch_size=cfg['batch_size'],
      num_workers=cfg['num_workers'],
      pin_memory=True,
      drop_last=True)

  return train_loader


def train_epoch(cfg, logger, train_loader, model, optimizer, criterion, epoch):
  # TODO(xwd): Add IoU calculator.
  batch_time = AverageMeter()
  data_time = AverageMeter()
  loss_meter = AverageMeter()

  model.train()
  end = time.time()
  max_iter = cfg['epochs'] * len(train_loader)

  for i, (image, target) in enumerate(train_loader):
    data_time.update(time.time() - end)
    # Avoid blocking during GPUs read data.
    image = image.cuda(non_blocking=True)
    target = target.type(torch.LongTensor).cuda(non_blocking=True)
    predict = model(image)
    # TODO(xwd): Parallel version should calculate loss before returning back to main processor.
    loss = criterion(predict, target)

    # Backprop.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    n = image.size(0)
    loss_meter.update(loss.item(), n)
    batch_time.update(time.time() - end)
    end = time.time()
    del loss

    # Update learning rate.
    # TODO(xwd): Add segment-updating learning rates.
    current_iter = epoch * len(train_loader) + i + 1
    current_lr = poly_learning_rate(
        cfg['base_lr'], current_iter, max_iter, power=cfg['power'])
    for param_group in optimizer.param_groups:
      param_group['lr'] = current_lr

    remain_iter = max_iter - current_iter
    remain_time = remain_iter * batch_time.avg
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    if (i + 1) % cfg['print_freq'] == 0:
      logger.info(
          f"Epoch: [{epoch+1}/{cfg['epochs']}][{i + 1}/{len(train_loader)}] "
          f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
          f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
          f"Remain {remain_time} "
          f"Loss {loss_meter.val:.4f} ")

  return loss_meter.avg


def main():
  logger, cfg, run_dir = get_logger_and_parser()

  # TODO(xwd): Adapt model type with config settings.
  model = network.get_model()
  if cfg['sync_bn']:
    logger.info('Convert batch norm layers to be sync.')
    model = convert_model(model).cuda()
  logger.info(
      f'Segmentation Network Total Params number: {count_model_param(model) / 1E6}M'
  )
  check_dir_exists(os.path.join(run_dir, 'model'))
  model_save_path = read_newest_model_path(os.path.join(run_dir, 'model'))

  # Set optimizer.
  # TODO(xwd): Set different learning rate for different parts of the model.
  optimizer = optim.SGD(
      model.parameters(),
      lr=cfg['base_lr'],
      momentum=cfg['momentum'],
      weight_decay=cfg['weight_decay'])

  # Resume.
  if cfg['resume'] is not None and os.path.isfile(cfg['resume']):
    checkpoint = torch.load(cfg['resume'])
    cfg['start_epoch'] = checkpoint['epoch']
    model = safe_loader(
        checkpoint['state_dict'],
        use_model='multi' if cfg['multi_gpu'] else 'single')
    optimizer.load_state_dict(checkpoint['optimizer'])
  else:
    if model_save_path is not None:
      checkpoint = torch.load(model_save_path)
      cfg['start_epoch'] = checkpoint['epoch']
      model = safe_loader(
          checkpoint['state_dict'],
          use_model='multi' if cfg['multi_gpu'] else 'single')
      optimizer.load_state_dict(checkpoint['optimizer'])

  # Set data loader.
  train_loader = get_dataloader(cfg)

  # Set criterion with ignore label if the dataset has.
  criterion = nn.CrossEntropyLoss(ignore_index=cfg['ignore_label'])

  # Training.
  for epoch in range(cfg['epochs']):
    # TODO(xwd): Add auxiliary loss as an optional loss for some specific type of models.
    epoch_log = epoch + 1
    _ = train_epoch(cfg, logger, train_loader, model, optimizer, criterion,
                    epoch)

    if (epoch_log % cfg['save_freq'] == 0):
      filename = os.path.join(model_save_path,
                              'checkpoint_' + str(epoch_log) + '.pth')
      logger.info('Saving checkpoint to: ' + filename)
      torch.save(
          {
              'epoch': epoch_log,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()
          }, filename)
      if epoch_log / cfg['save_freq'] > 2:
        deletename = os.path.join(
            model_save_path,
            'checkpoint_' + str(epoch_log - cfg['save_freq'] * 2) + '.pth')
        os.remove(deletename)


if __name__ == '__main__':
  main()
