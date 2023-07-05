import argparse
import pandas as pd
from pathlib import Path

from datasets import load_from_disk
import torch
from transformers.models.esm import EsmForSequenceClassification, EsmTokenizer
from transformers.utils import logging
from tqdm import tqdm

from rubisgen.util import read_fasta

parser = argparse.ArgumentParser()
parser.add_argument('--input_files', required=True, nargs='+', help='Input files')
parser.add_argument('--output_csv', required=True, help='Output csv')
parser.add_argument('--model_name_or_path', type=str, required=True, help='Checkpoint directory')
args = parser.parse_args()

tqdm.pandas()
logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def main():
    # Load input data
    df = []
    for filename in args.input_files:
        filename = Path(filename)
        if filename.suffix == '.csv':
            df_ = pd.read_csv(filename)
        elif filename.suffix in ('.fasta', '.fa'):
            records = read_fasta(filename, format='str')
            df_ = pd.DataFrame(list(records.items()), columns=['id', 'sequence'])
        elif filename.is_dir():
            dataset = load_from_disk(filename)
            df_ = pd.DataFrame(dataset['test'])
        else:
            raise ValueError(f'Unknown file format: {filename}')

        assert 'sequence' in df_.columns, f'No column sequence in {filename}'
        df.append(df_)
    df = pd.concat(df)

    # Load model
    model = EsmForSequenceClassification.from_pretrained(args.model_name_or_path)
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')

    # Evaluate sequences
    def _classify(row):
        inputs = tokenizer.batch_encode_plus([row['sequence']], return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=0)
        return probs[1].item()

    df['prob'] = df.progress_apply(_classify, axis=1)
    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
