#!/bin/sh

source scripts/tpu_commands.sh

# Remove any outdated info from known hosts.
for ip in $(tpu midGPT ips node3); do ssh-keygen -R $ip; done

tpu midGPT copy node3
tpu midGPT ssh node3 "pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
tpu midGPT ssh node3 "cd midGPT; pip install -r requirements.txt"

# Attach and mount PD that has dataset.
gcloud alpha compute tpus tpu-vm attach-disk node3 \
  --zone=europe-west4-a \
  --disk=owt \
  --mode=read-only

tpu midGPT ssh node3 "sudo mkdir -p /mnt/disks/persist"
tpu midGPT ssh node3 "sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist"
