from dataclasses import asdict
from datetime import datetime
import argparse
import os

import equinox as eqx
import gcsfs
import jax
import json

from src.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--rundir", type=str)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--multihost", action="store_true")
cmd_args = parser.parse_args()

if cmd_args.multihost:
    jax.distributed.initialize()
# load config from src.configs
config = getattr(
    __import__("src.configs", fromlist=[cmd_args.config]), cmd_args.config
).config
if cmd_args.rundir is not None:
    config.rundir = cmd_args.rundir
else:
    config.rundir = os.path.join(
        "outputs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
if cmd_args.debug:
    config.debug = True

print(f"Writing to {config.rundir}")
if config.rundir.startswith("gs://"):
    print("Using GCS filesystem")
    fs = gcsfs.GCSFileSystem()
    fopen = fs.open
else:
    print("Using local filesystem")
    fs = os
    fopen = open

# make sure the directory exists
fs.makedirs(config.rundir, exist_ok=True)

with fopen(os.path.join(config.rundir, "config.json"), "w") as f:
    f.write(json.dumps(asdict(config)))

eqx.tree_pprint(config)
train(config)
