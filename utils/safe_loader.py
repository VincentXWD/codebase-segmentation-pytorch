def single_gpu_load_multi_gpu_model(checkpoint):
  """ Use the multi-gpus trained model in a single gpu.
  """
  new_state_dict = {}
  for k, v in checkpoint.items():
    # remove prefix 'module'.
    name = k[7:]
    new_state_dict[name] = v
  # Use 'load_state_dict' to load parameters.
  return new_state_dict


def multi_gpu_load_single_gpu_model(checkpoint):
  """ Use the single gpu trained model in multi gpus.
  """
  new_state_dict = {}
  for k, v in checkpoint.items():
    # remove prefix 'module'.
    name = 'module.' + k
    new_state_dict[name] = v
  # Use 'load_state_dict' to load parameters.
  return new_state_dict


def safe_loader(checkpoint, use_model='multi'):
  """ Load checkpoint in a safe way. Helps check the model's training status.
  If use_model='multi', reads checkpoint into single gpu, else multi gpus.
  """
  for k, v in checkpoint.items():
    if use_model == 'multi':
      if 'module' in k:
        return checkpoint
      else:
        return multi_gpu_load_single_gpu_model(checkpoint)
    elif use_model == 'single':
      if 'module' in k:
        return single_gpu_load_multi_gpu_model(checkpoint)
      else:
        return checkpoint
    else:
      raise 'No such model, please check your parameters.'
