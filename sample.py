"""
Sample from a trained model
"""
import argparse
import os
import json
import pickle

from jax.experimental import mesh_utils
from src.model import GPT
from src.train import ExperimentConfig
import equinox as eqx
import gcsfs
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax  # type: ignore
import orbax.checkpoint as ocp
import tiktoken

from src.train import cast_pytree

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
cmd_args = parser.parse_args()

if cmd_args.ckpt_dir.startswith("gs://"):
    print("Using GCS filesystem for checkpoint")
    fs = gcsfs.GCSFileSystem()
    fopen = fs.open
else:
    print("Using local filesystem for checkpoint")
    fs = os
    fopen = open


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

    with fopen(json_path, "r") as f:
        json_string = f.read()
    return convert(json.loads(json_string), dataclass_type)


def generate(
    config, batched_model, idx, max_new_tokens, temperature=1.0, key=None
):
    block_size = config.model_config.block_size
    for _ in range(max_new_tokens):
        # take the final block_size tokens for conditioning, if the sequence is too long
        idx_cond = idx if idx.shape[1] <= block_size else idx[:, -block_size:]
        pluck_T = idx.shape[1] - 1
        if idx_cond.shape[1] < block_size:
            B, pad_T = idx_cond.shape[0], block_size - idx_cond.shape[1]
            padding = jnp.zeros((B, pad_T), dtype=idx_cond.dtype)
            idx_cond_new = jnp.concatenate([idx_cond, padding], axis=1)
        else:
            idx_cond_new = idx_cond
        # take the forward pass
        logits = batched_model(idx_cond_new)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, pluck_T, :] / temperature
        key, next_token_key = jrandom.split(key)
        # sample from the distribution
        idx_next = jax.random.categorical(
            next_token_key,
            logits,
            axis=1,
        ).reshape((idx.shape[0], 1))
        # append sampled index to the running sequence and continue
        idx = jnp.concatenate([idx, idx_next], axis=1)
    return idx


# load the model
config_path: str = os.path.join(cmd_args.ckpt_dir, "config.json")
config: ExperimentConfig = from_json(config_path, ExperimentConfig)
eqx.tree_pprint(config)

mngr = ocp.CheckpointManager(
    config.rundir,
    ocp.PyTreeCheckpointer(),
)
# model_leaves, _opt_state = mngr.restore(mngr.latest_step())
# model = GPT(config.model_config, key=jrandom.PRNGKey(0))
# model: GPT = jtu.tree_unflatten(jtu.tree_structure(model), model_leaves)

model = GPT(config.model_config, key=jrandom.PRNGKey(0))

# both of these are unused, but just for loading the checkpoint
scheduler = optax.warmup_cosine_decay_schedule(
    0,
    config.learning_rate,
    config.warmup_steps,
    config.lr_decay_steps,
    end_value=config.min_lr,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(b2=config.beta2),
    optax.add_decayed_weights(config.weight_decay / config.learning_rate),
    optax.scale_by_schedule(scheduler),
    optax.scale(-1),
)

#
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
ex_state = (jtu.tree_leaves(model), jtu.tree_leaves(opt_state))
ex_shardings = jtu.tree_map(lambda x: x.sharding if eqx.is_array(x) else None, ex_state)
restore_args = ocp.checkpoint_utils.construct_restore_args(ex_state, ex_shardings)
model_leaves, opt_state_leaves = mngr.restore(
    mngr.latest_step(), restore_kwargs={"restore_args": restore_args}
)
model = jtu.tree_unflatten(jtu.tree_structure(model), model_leaves)

# set up encoding/decoding
# the next several lines are copied directly from nanoGPT
# look for the meta pickle in case it is available in the dataset folder
# only for shakespeare_char
load_meta = False
meta_path = os.path.join(config.data_dir, "meta.pkl")
load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from LOCAL {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No LOCAL meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

model = cast_pytree(model, jnp.dtype(config.compute_dtype))

block_size = config.model_config.block_size
batched_model = eqx.filter_jit(jax.vmap(eqx.Partial(model, inference=True)))

# load the prompt
start = cmd_args.start
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()

key = jrandom.PRNGKey(0)

start_ids = encode(start if start != "" else "\n")
x = np.array([start_ids for _ in range(cmd_args.num_samples)])
devices = jax.devices()
mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), axis_names=("data",))
# TODO: currently replicating all data. Shard data properly.
data_sharding = NamedSharding(mesh, P(None, None))
x = jax.device_put(x, data_sharding)
jax.debug.visualize_array_sharding(x)
jax.debug.visualize_array_sharding(model.lm_head.weight_MxN)

print("generating samples...")
key, sample_key = jrandom.split(key)
y = generate(
    config,
    batched_model,
    x,
    cmd_args.max_new_tokens,
    temperature=cmd_args.temperature,
    key=sample_key,
)
samples = [decode(y[i].tolist()) for i in range(cmd_args.num_samples)]
for s in samples:
    print(s)
    print("---------------")
