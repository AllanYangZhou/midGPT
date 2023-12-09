# gcloud compute tpus tpu-vm scp setup.sh node3: --worker=all --zone=europe-west4-a
# gcloud compute tpus tpu-vm ssh node3 --zone=europe-west4-a --worker=all --command="chmod +x ./setup.sh; ./setup.sh"    

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install equinox==0.11.2 tiktoken==0.5.1 optax==0.1.7 tqdm==4.66.1 transformers==4.35.2 tiktoken==0.5.1 orbax==0.1.9 rich==13.7.0 tensorboard==2.15.1 tensorboardX==2.6.2.2
pip install tensorflow-cpu tensorboard-plugin-profile