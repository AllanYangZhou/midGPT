from src.train import ExperimentConfig
from src.model import GPTConfig

config = ExperimentConfig(
    rundir='',
    data_dir='data/openwebtext',
    learning_rate=1e-3,
    batch_size=128,
    warmup_steps=5000,
    min_lr=1e-5,
    lr_decay_steps=480_000,
    max_steps=480_000,
    beta2=0.95,
    weight_decay=1e-4,
    eval_interval=1000,
    policy='params=bfloat16,compute=bfloat16,output=bfloat16',
    g_accum_iters=4,
    shard_model=True,
    model_config=GPTConfig(
        block_size=1024, vocab_size=50304, n_layer=12, n_head=12,
        n_embd=768, dropout=0.0, bias=False,
    )
)
