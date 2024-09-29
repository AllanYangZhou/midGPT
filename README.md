# midGPT
A simple and hackable repository for experimenting on LLM pretraining, built using Jax+[Equinox](https://github.com/patrick-kidger/equinox). This codebase trains GPT-style decoder-only Transformers with billions of parameters on TPUs or GPUs.

MidGPT is inspired by [NanoGPT](https://github.com/karpathy/nanoGPT/), but supports FSDP across multiple devices and hosts for training larger models. It also includes some recent Transformer improvements: rotary embeddings, RMSNorm, QK-Layernorm, and independent weight decay, which can improve or stabilize training at larger scales.

Model code is in `src/model.py`, training code is in `src/train.py`. Experiments are configured in `src/configs/*.py`. Tested on Python **3.10.12**.

This project is supported by the [TPU Research Cloud](https://sites.research.google/trc/about/).

## Data preparation

As in nanoGPT, we support shakespeare_char (character-level prediction of Shakespeare texts) and openwebtext. The datasets are first processed into numpy memmapped `.bin` files:

```bash
cd data/openwebtext  # or data/shakespeare_char
python prepare.py
```

## Single host, multiple device setup
From a fresh Python 3.10+ virtualenv, [install Jax](https://jax.readthedocs.io/en/latest/installation.html) for your accelerator type, then `pip install -r requirements.txt`. To profile performance, also `pip install tensorflow-cpu tensorboard-plugin-profile`.

Start training:
```bash
export WANDB_API_KEY=<your key>
python launch.py --config=shakespeare_char
python launch.py --config=openwebtext  # 124M model
```

By default, this will create a timestamped rundir in `outputs/`. You can also manually specify `--rundir`, which is useful for resuming training:
```bash
# Create new run at rundir, or resume training if it already exists:
python launch.py --config=openwebtext --rundir=<rundir>
```

Add a `--debug` if you want to (1) enable jax profiler and (2) skip checkpoint saving.

## Multihost setup
Multihost training has only been tested on TPU slices (e.g., TPU v3-128), and we assume the dataset is openwebtext. Before starting, change the `tpu_project` and `tpu_zone` variables in `scripts/tpu_commands.sh` to your project ID and zone. Then, source the TPU commands:
```bash
source scripts/tpu_commands.sh
```


The data should be in a folder `openwebtext/` on a Google Cloud persistent disk, which will then be mounted to each host. Modify `scripts/setup.sh` with the correct zone and disk name, then:
```bash
./scripts/setup.sh <zone> <TPU name> <disk name> # after bringing up TPU slice
```

To start training a 1.5B model:
```bash
tpu midGPT ssh <TPU name> 'tmux new -d -s launch "WANDB_API_KEY=<your key> python ~/midGPT/launch.py --config=openwebtext_xl --multihost --rundir=gs://your_bucket_name/run_name"'
```

## Expected performance
The config `openwebtext.py` trains a 124M model analogous to nanoGPT, and should achieve ~2.80 val loss after all 60,000 steps. The config `openwebtext_xl.py` trains a 1.5B model, and should achieve a val loss ~2.42 after all 25,000 steps. On a TPU v3-128, the 1.5B model should take ~16.5 hours to train (throughput: ~444K tokens per second, MFU=47.8%).

## Acknowledgements
Compute was generously provided by the TPU Research Cloud (TRC).

* Tasks and data loading copied from [nanoGPT](https://github.com/karpathy/nanoGPT/)
* TPU shell commands adapted from [easyLM](https://github.com/young-geng/EasyLM)
*  Higher learning rates, independent weight decay, and QK-LayerNorm were adopted based on the results of [small-scale proxies](https://arxiv.org/abs/2309.14322)

## Citation
If you would like to cite this work:

```
@article{zhou2023midgpt,
    author={Allan Zhou and Nicholas C. Landolfi and Yiding Jiang},
    title={mid{GPT}: a simple and hackable repository for LLM pretraining},
    year={2023},
    url={https://github.com/AllanYangZhou/midGPT},
}
```
