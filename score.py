import argparse
import pandas as pd
from pathlib import Path
import os

from datasets import load_from_disk
import torch
from transformers.models.esm import EsmForSequenceClassification, EsmTokenizer
from transformers.utils import logging
from tqdm import tqdm

from rubisgen.modeling_progen import ProGenForCausalLM
from rubisgen.tokenization_progen import create_tokenizer
from rubisgen.util import read_fasta, write_fasta, sequence2record, string2hash
from rubisgen.alignment.blast import create_blastdb, blastp
from rubisgen.alignment.mmseqs import create_db, search, convertalis, parse_m8

blast_dir = Path('.blast')
os.environ['BLASTDB'] = str(blast_dir.resolve())

parser = argparse.ArgumentParser()
parser.add_argument('--input_files', required=True, nargs='+', help='Input files')
parser.add_argument('--output_csv', required=True, help='Output csv')
parser.add_argument('--generator_name_or_path', type=str, help='Generator checkpoint directory')
parser.add_argument('--discriminator_name_or_path', type=str, help='Discriminator checkpoint directory')
parser.add_argument('--target_fasta', default=None, type=str, help='Reference target fasta for mmseqs2')
parser.add_argument('--max_loss', default=None, type=float, help='Maximum loss for alignment')
parser.add_argument('--max_prob_disc', default=None, type=float, help='Maximum prob_disc for alignment')
parser.add_argument('--blast_num_threads', default=4, type=int, help='Number of threads in blastp')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing columns')
args = parser.parse_args()

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def main():
    # 0. load input data
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

    # 1. generator model
    if 'loss' in df.columns and not args.overwrite:
        sr_ = df['loss'].isna()
    else:
        sr_ = pd.Series([True] * len(df))

    if args.generator_name_or_path is not None and sr_.any():
        model = ProGenForCausalLM.from_pretrained(args.generator_name_or_path)
        tokenizer = create_tokenizer()

        def _score(row):
            input_ids = torch.tensor(
                tokenizer.encode('1' + row['sequence'] + '2').ids
            )
            input_ids = input_ids.unsqueeze(0).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            return outputs.loss.item()

        tqdm.pandas(desc=f'{args.input_files}: loss')
        df.loc[sr_, 'loss'] = df[sr_].progress_apply(_score, axis=1)

    # 2. discriminator model
    if 'prob_disc' in df.columns and not args.overwrite:
        sr_ = df['prob_disc'].isna()
    else:
        sr_ = pd.Series([True] * len(df))

    if args.discriminator_name_or_path is not None and sr_.any():
        model = EsmForSequenceClassification.from_pretrained(args.discriminator_name_or_path)
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')

        def _score(row):
            inputs = tokenizer.batch_encode_plus([row['sequence']], return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits[0], dim=0)
            return probs[1].item()

        tqdm.pandas(desc=f'{args.input_files}: prob_disc')
        df.loc[sr_, 'prob_disc'] = df[sr_].progress_apply(_score, axis=1)

    # 3. sequence alignment
    if 'pident' in df.columns and not args.overwrite:
        sr_ = df['pident'].isna()
    else:
        sr_ = pd.Series([True] * len(df))

    # restrict to loss and prob_disc criteria
    if args.max_loss is not None:
        sr_ = pd.concat([sr_, df['loss'] <= args.max_loss], axis=1)
        sr_ = sr_.all(axis=1)
    if args.max_prob_disc is not None:
        sr_ = pd.concat([sr_, df['prob_disc'] <= args.max_prob_disc], axis=1)
        sr_ = sr_.all(axis=1)

    if args.target_fasta is not None and sr_.any():
        # Mmseqs2 approach
        # mmseqs_dir = Path('.mmseqs')
        # mmseqs_dir.mkdir(exist_ok=True, parents=True)
        #
        # target_db_path = mmseqs_dir / 'target_db'
        # if not target_db_path.exists():
        #     create_db(args.target_fasta, target_db_path)
        #
        # def _score(row):
        #     tmp_dir = mmseqs_dir / 'tmp'
        #     tmp_dir.mkdir()
        #
        #     query_fasta = tmp_dir / 'query.fasta'
        #     query_db_path = tmp_dir / 'query_db'
        #
        #     query_records = [sequence2record(row['sequence'], row.get('id', None))]
        #     write_fasta(query_fasta, query_records)
        #     create_db(query_fasta, query_db_path)
        #     search(query_db_path, target_db_path, tmp_dir / 'searchDB', tmp_dir=tmp_dir, clean=False)
        #     convertalis(query_db_path, target_db_path, tmp_dir / 'searchDB', tmp_dir / 'search')
        #
        #     df_ = parse_m8(tmp_dir / 'search.m8')
        #     rmtree(tmp_dir)
        #
        #     if df_.empty:
        #         return 0
        #     return df_['pident'].max()
        #
        # df['pident'] = df.progress_apply(_score, axis=1)

        # Blast approach
        blast_dir.mkdir(exist_ok=True, parents=True)

        target_db_path = blast_dir / 'target_db'
        create_blastdb(args.target_fasta, target_db_path)
        target_sequences = read_fasta(args.target_fasta, format='str')

        def _score(row):
            record = sequence2record(row['sequence'], str(row.get('id', None)))
            best_hit = blastp(
                records=[record],
                db_path=target_db_path,
                blastp_dir=blast_dir,
                num_threads=args.num_threads,
            )

            alignment_title = best_hit['alignment_title']
            if alignment_title is None:
                best_hit['tseq'] = None
            else:
                best_hit['tseq'] = target_sequences[
                    alignment_title.replace('<unknown description>', '').split()[-1]
                ]

            return pd.Series(best_hit)

        # align entries
        tqdm.pandas(desc=f'{args.input_files}: blastp')
        df.loc[sr_, ['pident', 'alignment_title', 'tseq', 'score', 'evalue']] = df[sr_].progress_apply(_score, axis=1)

    df.to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
