from os import PathLike
from tokenizers import Tokenizer


def create_tokenizer(file: PathLike = 'data/models/tokenizer.json') -> Tokenizer:
    return Tokenizer.from_file(file)
