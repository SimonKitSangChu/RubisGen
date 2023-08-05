from pathlib import Path
from os import PathLike
import hashlib
import pickle
import json
from typing import *

from Bio import PDB
from Bio import SeqIO
from Bio.Seq import Seq
from datasets import Dataset, DatasetDict
import torch
from transformers.utils import logging
from transformers.utils.hub import TRANSFORMERS_CACHE

# logging config
logging.set_verbosity_info()
logger = logging.get_logger(__name__)


IdsLike = Union[List[int], torch.IntTensor, torch.LongTensor]
BatchIdsLike = Union[List[List[int]], torch.IntTensor, torch.LongTensor]


def write_fasta(fasta: PathLike, records: Iterable[SeqIO.SeqRecord]) -> None:
    with open(fasta, 'w') as handle:
        SeqIO.write(records, handle, 'fasta')


def read_fasta(fasta: PathLike, format: str = 'record', return_dict: bool = True) -> Union[Dict, List]:
    records = SeqIO.to_dict(SeqIO.parse(fasta, 'fasta'))
    if not return_dict:
        records = list(records.values())

    if format == 'record':
        return records
    elif format in ('sequence', 'str', 'string'):
        return records2sequences(records)
    else:
        raise ValueError(f'Unknown format: {format}')


def separate_fasta(fasta: PathLike, dest_dir: Optional[PathLike] = None, n_mer: int = 1) -> None:
    fasta = Path(fasta)
    if dest_dir is None:
        dest_dir = fasta.parent
    else:
        dest_dir = Path(dest_dir)

    records = read_fasta(fasta, format='record', return_dict=True)
    for k, record in records.items():
        write_fasta(
            dest_dir / f'{fasta.stem}_{k}.fasta',
            [record] * n_mer
        )


def read_json(json_file: PathLike) -> Dict[str, Any]:
    with open(json_file) as handle:
        return json.load(handle)


def write_json(data: Dict[str, Any], json_file: PathLike) -> None:
    with open(json_file, 'w') as handle:
        json.dump(data, handle, indent=2)


def read_pkl(pkl_file: PathLike, mode='rb') -> Any:
    with open(pkl_file, mode) as handle:
        return pickle.load(handle)


def write_pkl(data: Any, pkl_file: PathLike, mode='wb') -> None:
    with open(pkl_file, mode) as handle:
        pickle.dump(data, handle)


def record2sequence(record: SeqIO.SeqRecord) -> str:
    return str(record.seq)


def records2sequences(records: Iterable[SeqIO.SeqRecord]) -> List[str]:
    if isinstance(records, dict):
        return {k: record2sequence(v) for k, v in records.items()}
    else:
        return [record2sequence(record) for record in records]


def sequence2record(sequence: str, id_: str = '') -> SeqIO.SeqRecord:
    return SeqIO.SeqRecord(Seq(sequence), id=id_)


def sequences2records(sequences: Iterable[str]) -> Iterable[SeqIO.SeqRecord]:
    if issubclass(type(sequences), dict):
        return {str(id_): SeqIO.SeqRecord(Seq(sequence), id=str(id_)) for id_, sequence in sequences.items()}
    else:
        return [SeqIO.SeqRecord(Seq(sequence), id=str(id_)) for id_, sequence in enumerate(sequences)]


def spaceout(string: str) -> str:
    return ' '.join(string)


def unspace(string: str) -> str:
    return ''.join(string.split())


def string2hash(string: str, method: Callable = hashlib.sha256) -> str:
    hashname = string.encode()
    hashname = method(hashname)
    return hashname.hexdigest()


def clear_arrow_cache(cache_dir: Optional[PathLike] = None):
    if cache_dir is None:
        cache_dir = Path(TRANSFORMERS_CACHE)

    for arrow in cache_dir.glob('*.arrow'):
        arrow.unlink()


def train_val_test_split(
        dataset: Dataset,
        split: Tuple[float, float, float],
        seed: int = 42,
):
    eval_ratio = split[1] / (split[1] + split[2])
    dataset_dict = dataset.train_test_split(train_size=split[0], shuffle=True, seed=seed)
    split_dict = dataset_dict['test'].train_test_split(test_size=eval_ratio, seed=seed)
    dataset_dict['eval'] = split_dict['train']
    dataset_dict['test'] = split_dict['test']
    return DatasetDict(**dataset_dict)


def has_repeats(sequence: str, max_repeats: int = 10):
    for i in range(len(sequence) - max_repeats + 1):
        if all(sequence[i + j] == sequence[i] for j in range(1, max_repeats)):
            return True
    return False


def extract_bfactors(pdb_filename: PathLike, quiet: bool = True) -> List[float]:
    parser = PDB.PDBParser(QUIET=quiet)
    structure = parser.get_structure("structure", pdb_filename)

    b_factors = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    b_factors.append(ca_atom.get_bfactor())

    return b_factors
