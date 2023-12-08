import time
import os
import numpy as np
from src.train import get_batch

train_data = np.memmap(os.path.join('data/openwebtext', 'train.bin'), dtype=np.uint16, mode='r')
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
print(f"Batches per second: {100 / (end - start)}")