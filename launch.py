from dataclasses import asdict
from datetime import datetime
import argparse
import os

import equinox as eqx
import gcsfs
import jax
import json
import wandb
from jax.experimental.multihost_utils import sync_global_devices

from src.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--rundir", type=str)
parser.add_argument("--debug", action="store_true")
# TODO: we should just have separate launch scripts for these cases.
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
    assert not config.multihost, "Multihost must prespecify rundir."
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
    config.rundir = os.path.abspath(config.rundir)
    fs = os
    fopen = open

if jax.process_index() == 0:  # Wandb and config setup
    # make sure the directory exists
    fs.makedirs(config.rundir, exist_ok=True)

    config_dict = asdict(config)
    with fopen(os.path.join(config.rundir, "config.json"), "w") as f:
        f.write(json.dumps(config_dict))

    wandb_id_path = os.path.join(config.rundir, "wandb_id.txt")
    if fs.exists(wandb_id_path):
        with fopen(wandb_id_path, "r") as f:
            wandb_id = f.read()
    else:
        wandb_id = wandb.util.generate_id()
        with fopen(wandb_id_path, "w") as f:
            f.write(wandb_id)
    wandb.init(project="midgpt", id=wandb_id, resume="allow", config=config_dict)
if cmd_args.multihost:
    sync_global_devices("end_wandb_init")
eqx.tree_pprint(config)
train(config)
