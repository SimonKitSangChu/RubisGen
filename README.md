# RubisGen
This project is dedicated to RuBisCO sequence generation through decoder-only protein language model (pLM) from ProGen2 architecture/checkpoint.

# Installation
Create a conda environment for this project with `environment.yml`.
```
conda env create -n rubisgen --file=environment.yml
conda activate rubisgen
pip install -e .
```

# Usage
To build a dataset from fasta file,
```
python training.py --output_dir output --dataset_dir dataset --fasta input_fasta --num_proc=4 --build_dataset_only
```

After building the dataset, to train the model,
```
# check options
python training.py --help

# from scratch
python training.py \
    --output_dir output \
    --dataset_dir dataset \
    --local_batch_size=32 \
    --global_batch_size=2048 \
    --save_steps=20 \
    --logging_steps=20 \
    --scratch_model_config=config.json

# from model checkpoint
python training.py ... --model_checkpoint=ckpt_dir

# for multi-gpu training with deepspeed
deepspeed training.py ...
```

For model inference through generation,
```
# check options
python generate.py --help

# generate with default parameters
python generate.py --output_fasta dump.fasta --model_checkpoint=ckpt_dir

# generation with starting N-terminal sequence
python generate.py ... --start_tokens=ACDEF

# generate with pre-specified config in json
python generate.py ... --generation_config=config.json
```
