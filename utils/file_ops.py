import os
import shutil


def get_file_path(raw_input_dir):
  result = []
  for root, dirs, files in os.walk(raw_input_dir):
    for file in files:
      result.append(os.path.join(root, file))
  return result


def copy_file(input_path, output_path):
  if not os.path.isfile(input_path):
    raise "file {} doesnot exist".format(input_path)
  else:
    file_path, file_name = os.path.split(input_path)
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    shutil.copyfile(input_path, os.path.join(output_path, file_name))
    print("copy {} -> {}".format(input_path,
                                 os.path.join(output_path, file_name)))


def check_dir_exists(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)


def get_paths_with_prefix(paths, prefix):

  def filter_prefix(path):
    if prefix in path:
      return True
    return False

  return list(filter(filter_prefix, paths))


def read_csv(csv_path):
  with open(csv_path, 'r') as fp:
    lines = fp.readlines()
    lines = list(map(lambda each: each.strip(), lines))
    return lines
  return []


def write_csv(contents, file_name):
  with open(file_name, 'w') as fp:
    for path in contents:
      fp.write('{}\n'.format(path))
