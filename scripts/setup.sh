#!/bin/sh

source scripts/tpu_commands.sh

tpu midGPT copy node3
gcloud compute tpus tpu-vm ssh node3 --zone=europe-west4-a --worker=all --command="pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
gcloud compute tpus tpu-vm ssh node3 --zone=europe-west4-a --worker=all --command="cd midGPT; pip install -r requirements.txt"

# Attach and mount PD that has dataset.
gcloud alpha compute tpus tpu-vm attach-disk node3 \
  --zone=europe-west4-a \
  --disk=owt \
  --mode=read-only
sudo mkdir -p /mnt/disks/persist
sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist