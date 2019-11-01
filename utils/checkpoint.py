import os


def read_newest_model_path(model_dump_path):
  for _, _, files in os.walk(model_dump_path):
    files = list(filter(lambda path: 'checkpoint' in path, files))
    if len(files) == 0:
      return None
    files.sort()
    model_path = os.path.join(model_dump_path, files[-1])
    return model_path
