import argparse
import os
from datetime import datetime
from src.train import train


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--rundir', type=str)
cmd_args = parser.parse_args()

# load config from src.configs
config = getattr(__import__(
    'src.configs', fromlist=[cmd_args.config]), cmd_args.config).config
if cmd_args.rundir is not None:
    config.rundir = cmd_args.rundir
else:
    config.rundir = os.path.join(
        'outputs', datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
# make sure the directory exists
os.makedirs(config.rundir, exist_ok=True)
print(f"Writing to {config.rundir}")
train(config)