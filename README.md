# midGPT
A simple and hackable repository for experimenting on LLM pretraining, built using Jax+[Equinox](https://github.com/patrick-kidger/equinox). This codebase trains GPT-style decoder-only Transformers with billions of parameters on TPUs or GPUs.

MidGPT is inspired by [NanoGPT](https://github.com/karpathy/nanoGPT/), but supports FSDP-style sharding for training larger models and includes recent Transformer improvements: rotary embeddings, RMSNorm, QK-Layernorm, and independent weight decay.

Model code is in `src/model.py`, training code is in `src/train.py`. Experiments are configured in `src/configs/*.py`.

## Setup and start
Tested on Python **3.10.12**. From a fresh virtualenv, install Jax according to their [instructions](https://jax.readthedocs.io/en/latest/installation.html), then `pip install -r requirements.txt`. On TPU VMs, the Jax install is:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Start training:
```bash
python launch.py --config=shakespeare_char
python launch.py --config=openwebtext
# Resume training from existing rundir
python launch.py --config=openwebtext --rundir=<rundir>

# Launch tensorboard
tensorboard --logdir=<parent of rundir>
```

Add a `--debug` if you want to (1) enable jax profiler and (2) skip checkpoint saving.


## Debugging
* Testing parallelism in cpu: `JAX_PLATFORM_NAME=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 python train_shakespeare.py`.
* TB profiler: `pip install tensorflow-cpu tensorboard-plugin-profile`.

## Acknowledgements
Compute was generously provided by the TPU Research Cloud (TRC).

* Tasks and data loading copied from [nanoGPT](https://github.com/karpathy/nanoGPT/)
* TPU shell commands adapted from [easyLM](https://github.com/young-geng/EasyLM)
*  Higher learning rates, independent weight decay, and QK-LayerNorm were adopted based on the results of [small-scale proxies](https://arxiv.org/abs/2309.14322)

MidGPT was originally developed by Allan Zhou and Nick Landolfi, with helpful input from Yiding Jiang.