import numpy as np


def count_model_param(model):
  param_count = 0
  for param in model.parameters():
    param_count += param.view(-1).size()[0]
  return param_count


def show_model_parameters(net):
  for name, parameters in net.named_parameters():
    print(name, ':', parameters)


def show_model_parameters_std_and_mean(net):
  for name, parameters in net.named_parameters():
    print('[{}] std: {}, mean: {}'.format(
        name, np.std(parameters.cpu().detach().numpy()),
        np.mean(parameters.cpu().detach().numpy())))
