"""
Sample from a trained model
"""
import argparse
import os
import equinox as eqx
import numpy as np
import jax
from jax.experimental import mesh_utils
import jax.random as jrandom
import jmp
import json
import orbax.checkpoint as ocp
import pickle
from src.train import ExperimentConfig
from src.model import GPT
import tiktoken

jtu = jax.tree_util


NamedSharding, Mesh = jax.sharding.NamedSharding, jax.sharding.Mesh
P = jax.sharding.PartitionSpec

parser = argparse.ArgumentParser()
# outputs directory, e.g., outputs/2023-11-25-00-52-09
parser.add_argument("--ckpt_dir", type=str, required=True)
# start with "\n"... or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
parser.add_argument("--start", type=str, default="\n")
parser.add_argument("--num_samples", type=int, default=10)
parser.add_argument("--max_new_tokens", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0.8)
# retain only the top_k most likely tokens, clamp others to have 0 probability
parser.add_argument("--top_k", type=int, default=200)
cmd_args = parser.parse_args()


def from_json(json_path, dataclass_type):
    def convert(dict_or_list, dataclass_type):
        if isinstance(dict_or_list, dict):
            field_types = {
                f.name: f.type for f in dataclass_type.__dataclass_fields__.values()
            }
            return dataclass_type(
                **{k: convert(v, field_types[k]) for k, v in dict_or_list.items()}
            )
        elif isinstance(dict_or_list, list):
            return [convert(elem, dataclass_type.__args__[0]) for elem in dict_or_list]
        else:
            return dict_or_list

    with open(json_path, "r") as f:
        json_string = f.read()
    return convert(json.loads(json_string), dataclass_type)


# outputs/2023-11-25-00-52-09
# load the model
config_path: str = os.path.join(cmd_args.ckpt_dir, "config.json")
config: ExperimentConfig = from_json(config_path, ExperimentConfig)
eqx.tree_pprint(config)

devices = jax.devices()
print(devices)

options = ocp.CheckpointManagerOptions(
    max_to_keep=1, save_interval_steps=config.eval_interval
)
mngr = ocp.CheckpointManager(
    os.path.abspath(os.path.join(config.rundir, "ckpt_mngr")),
    ocp.PyTreeCheckpointer(),
    options=options,
)
model_leaves, _opt_state = mngr.restore(mngr.latest_step())
model = GPT(config.model_config, key=jrandom.PRNGKey(0))
model: GPT = jtu.tree_unflatten(jtu.tree_structure(model), model_leaves)

# set up encoding/decoding
# the next several lines are copied directly from nanoGPT
# look for the meta pickle in case it is available in the dataset folder
load_meta = False
meta_path = os.path.join(config.data_dir, "meta.pkl")
load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# load the prompt
start = cmd_args.start
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()

start_ids = encode(start)
x = np.array([start_ids])

mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), axis_names=("data",))
# TODO: currently replicating all data. Shard data properly.
data_sharding = NamedSharding(mesh, P(None, None))
x = jax.device_put(x, data_sharding)
jax.debug.visualize_array_sharding(x)
jax.debug.visualize_array_sharding(model.lm_head.weight)

print("generating samples...")
model = jmp.get_policy(config.policy).cast_to_compute(model)
key = jrandom.PRNGKey(0)
for _ in range(cmd_args.num_samples):
    key, sample_key = jrandom.split(key)
    y = model.generate(
        x,
        cmd_args.max_new_tokens,
        temperature=cmd_args.temperature,
        top_k=cmd_args.top_k,
        key=sample_key,
    )
    print(decode(y[0].tolist()))
    print("---------------")
