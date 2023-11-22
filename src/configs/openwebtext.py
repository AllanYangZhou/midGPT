from src.train import ExperimentConfig
from src.model import GPTConfig

config = ExperimentConfig(
    rundir='',
    data_dir='data/openwebtext',
    learning_rate=6e-4,
    # Ideal effective batch size: 480
    batch_size=120,
    warmup_steps=2000,
    min_lr=6e-5,
    lr_decay_steps=600_000,
    max_steps=600_000,
    beta2=0.95,
    weight_decay=0.1,
    eval_interval=1000,
    policy='params=float32,compute=bfloat16,output=bfloat16',
    g_accum_steps=4,
    shard_model=True,
    model_config=GPTConfig(
        block_size=1024, vocab_size=50304, n_layer=12, n_head=12,
        n_embd=768, dropout=0.0, bias=False,
    )
)
