from src.train import ExperimentConfig
from src.model import GPTConfig

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
    weight_decay=1e-4,
    eval_interval=2000,
    compute_dtype='bfloat16',
    param_dtype='float32',
    g_accum_iters=1,
    shard_model=False,
    model_config=GPTConfig(
        block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.2),
)
