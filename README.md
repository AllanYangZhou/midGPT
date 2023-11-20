# NanoGPT EQX
A simple and hackable repository for training medium-sized (hopefully, even somewhat large) GPTs. Like [NanoGPT](https://github.com/karpathy/nanoGPT/), but using Jax+[Equinox](https://github.com/patrick-kidger/equinox) instead of PyTorch. 

Model code is in `src/model.py`, training code is in `src/train.py`. Experiments are configured and launched by `train_<dset>.py` scripts (e.g., `train_owt.py`).

## Setup and start
From a fresh virtualenv, install Jax according to their [instructions](https://jax.readthedocs.io/en/latest/installation.html), then `pip install -r requirements.txt`. Tested on Python 3.11.0, but Python 3.9+ probably works.

```bash
python train_shakespeare.py  # character-level shakespeare
python train_owt.py  # openwebtext
```

## Feature list

 - [x] Basic training on OWT
 - [x] Mixed precision
 - [x] Data parallel training
 - [ ] FSDP
 - [ ] Grad accumulation
 - [ ] Checkpointing and resuming
 - [ ] Logging


## Debugging
* Debugging parallelism in cpu: `JAX_PLATFORM_NAME=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 python train_shakespeare.py`.