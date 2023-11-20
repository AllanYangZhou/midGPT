import argparse
import os
from datetime import datetime
from src.train import ExperimentConfig, train
from src.model import GPTConfig
import jmp


config = ExperimentConfig(
    rundir='',
    data_dir='data/shakespeare_char',
    learning_rate=1e-3,
    batch_size=64,
    warmup_steps=100,
    min_lr=1e-4,
    lr_decay_steps=5000,
    max_steps=5000,
    beta2=0.99,
    weight_decay=0.1,
    eval_interval=2000,
    policy=jmp.get_policy("params=float32,compute=float32,output=float32"),
    model_config=GPTConfig(
        block_size=256, vocab_size=65, n_layer=6, n_head=6,
        n_embd=384, dropout=0.2, bias=False,
    )
)

parser = argparse.ArgumentParser()
parser.add_argument('--rundir', type=str)
cmd_args = parser.parse_args()
if cmd_args.rundir is not None:
    config.rundir = cmd_args.rundir
else:
    config.rundir = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# make sure the directory exists
os.makedirs(config.rundir, exist_ok=True)
print(f"Writing to {config.rundir}")
train(config)