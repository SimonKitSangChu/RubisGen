from os import PathLike
from tokenizers import Tokenizer

def create_tokenizer(file: PathLike = 'data/models/tokenizer.json') -> Tokenizer:
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())
