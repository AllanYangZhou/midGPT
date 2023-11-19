from src.train import ExperimentConfig, train
from src.model import GPTConfig
import jmp


config = ExperimentConfig(
    data_dir='/scr/ayz/nano/openwebtext',
    learning_rate=6e-4,
    batch_size=12,
    warmup_steps=2000,
    min_lr=6e-5,
    lr_decay_steps=600_000,
    max_steps=600_000,
    beta2=0.95,
    weight_decay=0.1,
    eval_interval=1000,
    policy=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
    model_config=GPTConfig(
        block_size=1024, vocab_size=50304, n_layer=12, n_head=12,
        n_embd=768, dropout=0.0, bias=False,
    )
)

train(config)