import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import time
import os
import numpy as np
import jax
from src.train import get_batch

start = time.time()
train_data = np.memmap(os.path.join('/mnt/disks/persist/openwebtext', 'train.bin'), dtype=np.uint16, mode='r').copy()
print(f"Worker {jax.process_index()}; Time to load train.bin: {time.time() - start}")
x_GxBxD, y_GxBxD = get_batch(
    train_data, 1024, 128, 4
)

# time how long it takes to get 100 batches
start = time.time()
for i in range(100):
    x_GxBxD, y_GxBxD = get_batch(
        train_data, 1024, 128, 4
    )
end = time.time()
print(f"Worker {jax.process_index()}; Batches per second: {100 / (end - start)}")