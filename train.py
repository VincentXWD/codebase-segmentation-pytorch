"""
@author: Wendong Xu
@contact: kirai.wendong@gmail.com
@file: train.py
@time: 2019-11-01 19:17
@desc: Use CUDA by default. Support distributed training. Use NCCL as backend.
"""
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
import torch.distributed as dist

import network
import utils.transforms as transform

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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


cfg, logger = None, None


def main_process():
    return cfg['local_rank'] == 0


def get_logger_and_parser():
  global cfg, logger

  parser = argparse.ArgumentParser(description='config')
  parser.add_argument(
      '--config',
      type=str,
      default='config/cityscapes.yaml',
      help='Configuration file to use',
  )
  parser.add_argument(
      '--local_rank',
      type=int,
      default=0,
      help='Local rank for distributed training',
  )

  args = parser.parse_args()

  assert args.config is not None
  cfg = config.load_cfg_from_cfg_file(args.config)
  args_dict = dict()
  for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
  cfg.update(args_dict)

  run_dir = os.path.join('runs',
                         os.path.basename(args.config)[:-5], cfg['exp_name'])

  if main_process():
    check_dir_exists(run_dir)
    run_id = str(int(time.time()))
    logger = get_logger(run_dir, run_id, 'train')
    logger.info('RUNDIR: {}'.format(run_dir))
    shutil.copy(args.config, run_dir)
  else:
    logger = None

  try:
    cfg['world_size'] = int(os.environ['WORLD_SIZE'])
  except:
    pass

  return logger, cfg, run_dir


def get_dataloader():
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

  # Use data sampler to make sure each GPU loads specific parts of dataset to avoid data reduntant.
  train_sampler = DistributedSampler(train_data)

  train_loader = DataLoader(
      train_data,
      batch_size=cfg['batch_size'] // cfg['world_size'],
      shuffle=(train_sampler is None),
      num_workers=4,
      pin_memory=True,
      sampler=train_sampler,
      drop_last=True)

  return train_loader, train_sampler


def train_epoch(train_loader, model, optimizer, epoch):
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
    loss = model(image, target)

    # Backprop.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    n = image.size(0)
    if cfg['multiprocessing_distributed']:
      loss = loss.detach() * n
      count = target.new_tensor([n], dtype=torch.long)
      dist.all_reduce(loss)
      dist.all_reduce(count)
      n = count.item()
      loss = loss / n
    else:
      loss = torch.mean(loss)

    loss_meter.update(loss.item(), n)
    batch_time.update(time.time() - end)
    end = time.time()
    del loss

    # TODO(xwd): Add segment-updating learning rates.
    # Update learning rate.
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
    if main_process() and (i + 1) % cfg['print_freq'] == 0:
      logger.info(
          f"Epoch: [{epoch+1}/{cfg['epochs']}][{i + 1}/{len(train_loader)}] "
          f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
          f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
          f"Remain {remain_time} "
          f"Loss {loss_meter.val:.4f} ")


def main():
  global logger, cfg

  logger, cfg, run_dir = get_logger_and_parser()

  # Settings for distributed training.
  torch.cuda.set_device(cfg['local_rank'])
  dist.init_process_group('nccl', init_method='env://')

  # Set model & criterion.
  # TODO(xwd): Adapt model type with config settings.
  model_path = os.path.join(run_dir, 'model')
  criterion = nn.CrossEntropyLoss(ignore_index=cfg['ignore_label'])
  model = network.get_model(criterion)

  if cfg['multi_gpu']:
    if cfg['sync_bn']:
      if main_process():
        logger.info('Convert batch norm layers to be sync.')
      # Manually-implemented sync_bn is deprecated. Use pytorch implementation instead.
      # model = convert_model(model)
      model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()

    model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg['local_rank']], output_device=cfg['local_rank'])

  if main_process():
    logger.info(
        f'Segmentation Network Total Params number: {count_model_param(model) / 1E6}M'
    )

  check_dir_exists(model_path)

  # Set optimizer.
  # TODO(xwd): Set different learning rate for different parts of the model.
  optimizer = optim.SGD(
      model.parameters(),
      lr=cfg['base_lr'],
      momentum=cfg['momentum'],
      weight_decay=cfg['weight_decay'])

  # Resume.
  # TODO(xwd): This method will consume more GPU memory in main GPU. Try to reduce it.
  if cfg['resume'] is not None and os.path.isfile(cfg['resume']):
    checkpoint = torch.load(cfg['resume'])
    cfg['start_epoch'] = checkpoint['epoch']
    model.load_state_dict(
        safe_loader(
            checkpoint['state_dict'],
            use_model='multi' if cfg['multi_gpu'] else 'single'))
    optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint
  else:
    model_save_path = read_newest_model_path(model_path)
    if model_save_path is not None:
      checkpoint = torch.load(model_save_path)
      cfg['start_epoch'] = checkpoint['epoch']
      model.load_state_dict(
          safe_loader(
              checkpoint['state_dict'],
              use_model='multi' if cfg['multi_gpu'] else 'single'))
      optimizer.load_state_dict(checkpoint['optimizer'])
      if main_process():
        logger.info(
            f"Pretrained checkpoint loaded. Start from epoch {cfg['start_epoch']}."
        )
      del checkpoint

  # Set data loader & sampler.
  train_loader, train_sampler = get_dataloader()

  # Training.
  for epoch in range(cfg['start_epoch'], cfg['epochs']):
    # TODO(xwd): Add auxiliary loss as an optional loss for some specific type of models.
    if cfg['multiprocessing_distributed']:
      train_sampler.set_epoch(epoch)

    epoch_log = epoch + 1
    train_epoch(train_loader, model, optimizer, epoch)

    if main_process() and (epoch_log % cfg['save_freq'] == 0):
      logger.info(model_path, f'checkpoint_{epoch_log}.pth')
      filename = os.path.join(model_path, 'checkpoint_{}.pth'.format(epoch_log))
      logger.info(f'Saving checkpoint to: {filename}')
      torch.save(
          {
            'epoch': epoch_log,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
          }, filename)
      if epoch_log / cfg['save_freq'] > 2:
        deletename = os.path.join(
            model_path, f"checkpoint_{epoch_log - cfg['save_freq'] * 2}.pth")
        if os.path.exists(deletename):
          os.remove(deletename)


if __name__ == '__main__':
  main()
