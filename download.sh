# checkpoint
model=progen2-xlarge
mkdir -p data/models
wget -P data/models/checkpoints/${model} https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz
tar -xvf data/models/checkpoints/${model}/${model}.tar.gz -C data/models/checkpoints/${model}/
