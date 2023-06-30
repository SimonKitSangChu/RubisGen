import argparse
from datetime import datetime
from pathlib import Path
import sys

from datasets import load_from_disk, Dataset
import torch
from tokenizers import Tokenizer
from transformers.models.esm import EsmTokenizer
from transformers.utils import logging

from rubisgen.util import write_json, unspace, read_fasta
from rubisgen.data import train_val_test_split, clear_arrow_cache
from rubisgen.tokenization_progen import create_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default=None, required=True, help='Dataset directory')
parser.add_argument('--fasta', default=None, required=True, help='Fasta file')
parser.add_argument('--generated_fasta', default=None, nargs='+', help='Generated fasta file(s)')
parser.add_argument('--dataset_n_chunks', default=1, type=int, help='Build dataset in chunks')
parser.add_argument('--num_proc', default=None, type=int, help='num_proc for building dataset')
parser.add_argument('--test_ratio', default=0.02, type=float, help='Validation/Test ratio to training set')
parser.add_argument('--seed', default=42, type=int, help='Random seed')
parser.add_argument('--min_length', default=100, type=int, help='Min length')
parser.add_argument('--max_length', default=600, type=int, help='Max length')
args, unk = parser.parse_known_args()

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def main():
    # 0. global config
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(vars(args), dataset_dir / 'args.json')

    if unk:
        logger.warning(f'unknown args: {unk}')
    write_json(unk, dataset_dir / 'unk.json')

    try:  # avoid overwriting existing dataset
        load_from_disk(dataset_dir)
        logger.info(f'Found dataset in {dataset_dir}')
        sys.exit(0)
    except FileNotFoundError:
        pass

    if args.generated_fasta is None:
        logger.info('Since --generated_fasta is None, the dataset will be a single sequence dataset under ProGen2 '
                    'tokenizer.')
        is_discriminator = False
    else:
        logger.info('Since --generated_fasta is not None, the dataset will be a binary classification dataset under '
                    'ESM tokenizer.')
        is_discriminator = True

    # 1. create dataset
    logger.info(f'Creating dataset in {dataset_dir}')

    sequences = read_fasta(args.fasta, format='sequence', return_dict=True)
    if args.generated_fasta is None:
        dataset = Dataset.from_dict({
            'id': list(sequences.keys()),
            'sequence': list(sequences.values())
        })
    else:
        generated_ids, generated_sequences = [], []
        for generated_fasta in args.generated_fasta:
            sequences_ = read_fasta(generated_fasta, format='sequence', return_dict=True)
            generated_ids.extend(list(sequences_.keys()))
            generated_sequences.extend(list(sequences_.values()))

        dataset = Dataset.from_dict({
            'id': list(sequences.keys()) + generated_ids,
            'sequence': list(sequences.values()) + generated_sequences,
            'labels': [0] * len(sequences) + [1] * len(generated_sequences)
        })

    # 2. processing dataset
    logger.info(f'Processing dataset in {dataset_dir}')

    min_length = args.min_length
    max_length = args.max_length

    if is_discriminator:
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')
    else:
        tokenizer = create_tokenizer()

    def filter_function(examples):
        sequence = unspace(examples['sequence'])
        if 'X' in sequence:
            return False
        if len(sequence) < min_length:
            return False
        elif max_length is not None and len(sequence) > max_length:
            return False

        return True

    if issubclass(tokenizer.__class__, Tokenizer):
        def tokenize_function(examples):
            sequence = '1' + unspace(examples['sequence']).lstrip('1').strip('2') + '2'
            return {'input_ids': tokenizer.encode(sequence).ids}
    else:
        def tokenize_function(examples):
            sequence = unspace(examples['sequence'])
            return {'input_ids': tokenizer.encode(sequence)}

    dataset = dataset.filter(filter_function, batched=False)
    dataset = dataset.map(
        tokenize_function,
        batched=False,
        num_proc=torch.get_num_threads() if args.num_proc is None else args.num_proc,
    )
    dataset = train_val_test_split(
        dataset,
        split=(1 - 2 * args.test_ratio, args.test_ratio, args.test_ratio),
        seed=args.seed
    )

    # 3. save dataset
    logger.info(f'Saving dataset in {dataset_dir}')

    dataset.save_to_disk(dataset_dir)
    clear_arrow_cache()
    meta_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'fasta': args.fasta,
        'generated_fasta': args.generated_fasta,
    }
    write_json(meta_data, dataset_dir / 'metadata.json')


if __name__ == '__main__':
    main()
