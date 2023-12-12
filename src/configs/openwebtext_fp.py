from src.train import ExperimentConfig
from src.model import GPTConfig

config = ExperimentConfig(
    rundir='',
    data_dir='/mnt/disks/persist/openwebtext',
    learning_rate=6e-4,
    batch_size=512,
    warmup_steps=5_000,
    min_lr=6e-5,
    lr_decay_steps=60_000,
    max_steps=60_000,
    beta2=0.95,
    weight_decay=6e-5,
    eval_interval=1000,
    compute_dtype='bfloat16',
    param_dtype='float32',
    g_accum_iters=1,
    shard_model=False,
    model_config=GPTConfig(
        block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.0)
)
