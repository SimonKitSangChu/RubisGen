from datetime import datetime
from pathlib import Path
from os import PathLike
import shutil
from typing import Optional, Any, Dict, Tuple, Iterable, List, Union

from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.utils import logging
import torch

from protldm.util import read_fasta, spaceout, unspace, write_json, clear_arrow_cache, train_val_test_split

logger = logging.get_logger(__name__)


def create_test_dataset(
        tokenizer: PreTrainedTokenizer,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        max_length: int = 512,
):
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {'padding': 'longest', 'truncation': True, 'max_length': 2048}

    sequences = [
        'AAAA' * max_length,
        'ATCG' * max_length,
    ]
    dataset = Dataset.from_dict({'sequence': sequences})

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        sequence = spaceout(unspace(examples['sequence']))
        return tokenizer(sequence, **tokenizer_kwargs)

    dataset = dataset.map(tokenize_function, batched=False, )
    dataset_dict = DatasetDict(train=dataset, eval=dataset, test=dataset)
    return dataset_dict


def create_dataset(
        dataset_dir: Optional[PathLike],
        tokenizer: PreTrainedTokenizer,
        fasta: PathLike = 'data/uniprot/uniref50.fasta',
        sequences: Optional[Iterable[str]] = None,
        split: Optional[Tuple[float, float, float]] = (0.99, 0.005, 0.005),
        filtering_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
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
            tokenizer_kwargs = {'padding': 'longest', 'truncation': True, 'max_length': 2048}

        if filtering_kwargs is None:
            filtering_kwargs = {}
        min_length = filtering_kwargs.get('min_length', 8)
        max_length = filtering_kwargs.get('max_length', None)

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
            sequence = spaceout(unspace(examples['sequence']))
            return tokenizer(sequence, **tokenizer_kwargs)

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
            if dataset_dir:
                dataset.save_to_disk(dataset_dir)
                clear_arrow_cache()
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

        if dataset_dir:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'fasta': fasta,
                'tokenizer': tokenizer.__class__.__name__,
                'tokenizer_kwargs': tokenizer_kwargs,
            }
            write_json(metadata, dataset_dir / 'metadata.json')

    if split is None:
        return dataset
    else:
        return train_val_test_split(dataset, split=split, seed=seed)


class DataCollatorForProtT5(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizer, mlm_probability: float = 0.15, mlm: bool = True,
                 skip_special_tokens: bool = False):
        super().__init__(tokenizer, mlm_probability=mlm_probability, mlm=mlm)
        self.skip_special_tokens = skip_special_tokens

    def __call__(self, features, return_tensors='pt'):
        if return_tensors != 'pt':
            raise NotImplementedError('Only PyTorch tensors are supported.')

        return super().__call__(features, return_tensors=return_tensors)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        if self.skip_special_tokens:
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices & ~special_tokens_mask
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced & ~special_tokens_mask
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

