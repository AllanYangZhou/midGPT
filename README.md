## Setup and start
Install Jax according to their instructions, then `pip install -r requirements.txt`. Tested on Python 3.11.0, but Python 3.9+ probably works.

```bash
python train_shakespeare.py  # character-level shakespeare
python train_owt.py  # openwebtext
```

## Feature list

 - [x] Basic training on OWT
 - [x] Mixed precision
 - [ ] Grad accumulation
 - [ ] Checkpointing and resuming
 - [ ] Logging
 - [ ] Distributed training