from src.train import ExperimentConfig
from src.model import GPTConfig

config = ExperimentConfig(
    rundir='',
    data_dir='/mnt/disks/persist/openwebtext',
    learning_rate=1e-3,
    batch_size=1024,
    warmup_steps=2500,
    min_lr=1e-5,
    lr_decay_steps=25_000,
    max_steps=25_000,
    beta2=0.95,
    weight_decay=1e-4,
    eval_interval=1000,
    compute_dtype='bfloat16',
    param_dtype='float32',
    g_accum_iters=1,
    shard_model=True,
    model_config=GPTConfig(
        block_size=1024, vocab_size=50304, n_layer=24, n_head=16, n_embd=2048, dropout=0.0)
)
