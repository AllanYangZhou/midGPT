# midGPT
A simple and hackable repository for training mid-sized GPTs (language models).  Built using Jax+[Equinox](https://github.com/patrick-kidger/equinox). Directly inspired by [NanoGPT](https://github.com/karpathy/nanoGPT/).

Model code is in `src/model.py`, training code is in `src/train.py`. Experiments are configured in `src/configs/*.py`.

## Setup and start
From a fresh virtualenv, install Jax according to their [instructions](https://jax.readthedocs.io/en/latest/installation.html), then `pip install -r requirements.txt`. Tested on Python 3.11.0, but Python 3.9+ probably works.

```bash
python launch.py  --config=shakespeare_char
python launch.py --config=openwebtext
```

## Feature list

 - [x] Basic training on OWT
 - [x] Mixed precision
 - [x] Data parallel training
 - [x] FSDP
 - [x] Checkpointing and resuming
 - [x] Logging
 - [ ] Grad accumulation


## Debugging
* Debugging parallelism in cpu: `JAX_PLATFORM_NAME=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 python train_shakespeare.py`.