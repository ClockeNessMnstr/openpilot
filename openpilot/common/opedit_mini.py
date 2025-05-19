import os
import json

from openpilot.common.basedir import BASEDIR

BASEDIR = os.path.dirname(BASEDIR)
PARAMS_DIR = os.path.join(BASEDIR, 'params', 'community')

def write_param(key, value):
  if not os.path.exists(PARAMS_DIR):
    os.makedirs(PARAMS_DIR)
  param_path = os.path.join(PARAMS_DIR, key)
  with open(param_path, "w") as f:
    f.write(json.dumps(value))

def read_param(key):  # Returns None, False if a json error occurs
  try:
    with open(os.path.join(PARAMS_DIR, key), 'r') as f:
      value = json.loads(f.read())
    return value, True
  except json.decoder.JSONDecodeError:
    return (None, False)
  except FileNotFoundError:
    return (None, False)
