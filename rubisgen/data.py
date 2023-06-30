from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from os import PathLike
import shutil
from typing import Optional, Any, Dict, Tuple, Iterable, List

from tokenizers import Tokenizer
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import PreTrainedTokenizer
from transformers.utils import logging
import torch

from .util import read_fasta, unspace, write_json, clear_arrow_cache, train_val_test_split

logger = logging.get_logger(__name__)


def create_dataset(
        dataset_dir: Optional[PathLike],
        tokenizer: PreTrainedTokenizer,
        fasta: PathLike = 'data/uniprot/uniref50.fasta',
        sequences: Optional[Iterable[str]] = None,
        split: Optional[Tuple[float, float, float]] = (0.99, 0.005, 0.005),
        filtering_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        min_length: int = 100,
        max_length: int = 600,
        num_proc: Optional[int] = None,
        seed: int = 42,
        n_chunks: int = -1,
):
    dataset = None

    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir)
        if dataset_dir.exists():
            try:
                dataset = load_from_disk(dataset_dir)
            except FileNotFoundError:
                pass

    if dataset is None:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        if filtering_kwargs is None:
            filtering_kwargs = {}
        min_length = filtering_kwargs.get('min_length', min_length)
        max_length = filtering_kwargs.get('max_length', max_length)

        def filter_function(examples: Dict[str, Any]) -> Dict[str, Any]:
            sequence = unspace(examples['sequence'])
            if 'X' in sequence:
                return False
            if len(sequence) < min_length:
                return False
            elif max_length is not None and len(sequence) > max_length:
                return False

            return True

        def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
            sequence = '1' + unspace(examples['sequence']).lstrip('1').rstrip('2') + '2'
            if issubclass(tokenizer.__class__, Tokenizer):
                input_ids = tokenizer.encode(sequence, **tokenizer_kwargs).ids
            else:
                input_ids = tokenizer(sequence, **tokenizer_kwargs)
            return {'input_ids': input_ids}

        if sequences is None:
            sequences = read_fasta(fasta, format='sequence', return_dict=True)

        if n_chunks <= 1:
            dataset = Dataset.from_dict({'id': list(sequences.keys()), 'sequence': list(sequences.values())})
            dataset = dataset.filter(filter_function, batched=False)
            dataset = dataset.map(
                tokenize_function,
                batched=False,
                num_proc=torch.get_num_threads() if num_proc is None else num_proc,
            )
        else:
            keys = list(sequences.keys())
            sequences_list = list(sequences.values())
            for i in range(0, n_chunks):
                logger.info(f'Process n_chunks [{i+1}|{n_chunks}]')
                chunk_dir = dataset_dir / f'chunk_{i}'
                if not chunk_dir.exists():
                    dataset_chunk = Dataset.from_dict({
                        'id': keys[i::n_chunks],
                        'sequence': sequences_list[i::n_chunks],
                    })
                    dataset_chunk = dataset_chunk.filter(filter_function, batched=False)
                    dataset_chunk = dataset_chunk.map(
                        tokenize_function,
                        batched=False,
                        num_proc=torch.get_num_threads() if num_proc is None else num_proc,
                        desc=f'map [{i+1}|{n_chunks} chunks]',
                    )
                    dataset_chunk.save_to_disk(dataset_dir / f'chunk_{i}')
                    clear_arrow_cache()

            dataset = []
            chunk_dirs = [chunk_dir for chunk_dir in dataset_dir.glob('chunk_*') if chunk_dir.is_dir()]
            for chunk_dir in chunk_dirs:
                dataset_chunk = load_from_disk(chunk_dir)
                dataset.append(dataset_chunk)

            dataset = concatenate_datasets(dataset)
            dataset.save_to_disk(dataset_dir)
            for chunk_dir in chunk_dirs:
                shutil.rmtree(chunk_dir)
            dataset = load_from_disk(dataset_dir)  # reload to reset cache

    if split is not None:
        dataset = train_val_test_split(dataset, split=split, seed=seed)

    if dataset_dir:
        dataset.save_to_disk(dataset_dir)
        clear_arrow_cache()
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'fasta': fasta,
        }
        write_json(metadata, dataset_dir / 'metadata.json')

    return dataset


@dataclass
class DataCollatorWithPadding:
    tokenizer: Tokenizer
    pad_token_id: int = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(len(x['input_ids']) for x in features)

        input_ids = []
        attention_mask = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.pad_token_id] * (max_length - len(feature['input_ids'])))
            attention_mask.append([1] * len(feature['input_ids']) + [0] * (max_length - len(feature['input_ids'])))

        if 'labels' in features[0]:
            labels = []
            for feature in features:
                labels_ = feature['labels']
                if isinstance(labels_, list):
                    labels_ = labels_ + [self.pad_token_id] * (max_length - len(labels_))
                else:
                    labels_ = [labels_]
                labels.append(labels_)
        else:
            labels = input_ids

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
