#!/bin/sh

source scripts/tpu_commands.sh

# Remove any outdated info from known hosts.
for ip in $(tpu midGPT ips $2); do ssh-keygen -R $ip; done

tpu midGPT copy $2
tpu midGPT ssh $2 "pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
tpu midGPT ssh $2 "cd midGPT; pip install -r requirements.txt"

# Attach and mount PD that has dataset.
gcloud alpha compute tpus tpu-vm attach-disk $2 \
  --zone=$1 \
  --disk=$3 \
  --mode=read-only

tpu midGPT ssh $2 "sudo mkdir -p /mnt/disks/persist"
tpu midGPT ssh $2 "sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist"
