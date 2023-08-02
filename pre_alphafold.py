import argparse
import pandas as pd
from pathlib import Path
import os

from transformers.utils import logging
from tqdm import tqdm

from rubisgen.util import read_fasta, write_fasta, sequence2record, sequences2records, has_repeats, records2sequences
from rubisgen.alignment.mmseqs import easy_cluster
from rubisgen.alignment.blast import blastp

parser = argparse.ArgumentParser()
parser.add_argument('--input_dirs', required=True, nargs='+',
                    help='Direcotries containing input csv(s)')
parser.add_argument('--max_loss', type=float, default=None, help='Maximum loss')
parser.add_argument('--max_prob_disc', type=float, default=None, help='Maximum discriminator probability')
parser.add_argument('--max_repeats', type=int, default=10, help='Maximum number of repeats')
parser.add_argument('--target_db_path', type=str, default=None, help='Path to blast database')
args = parser.parse_args()

tqdm.pandas()
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

blast_dir = Path('.blast')
os.environ['BLASTDB'] = blast_dir


def main():
    # 1. load input csv(s)
    df = []
    for input_dir in args.input_dirs:
        input_dir = Path(input_dir)

        if input_dir.is_file() and input_dir.suffix == '.csv':
            csv_list = [input_dir]
        else:
            csv_list = input_dir.glob('*.csv')

        for csv in csv_list:
            df_ = pd.read_csv(csv)
            assert 'sequence' in df_.columns, f'No column sequence in {csv}'
            df.append(df_)

    df = pd.concat(df)

    # 2. filtering
    pident_bins = [0, 30, 40, 50, 60, 70, 100]
    pident_labels = ['0-30', '30-40', '40-50', '50-60', '60-70', '70-100']
    df['pident_bins'] = pd.cut(df['pident'], bins=pident_bins, labels=pident_labels, right=False)

    sequences = set()
    for pident_bin, df_ in df.groupby('pident_bins'):
        # filter by criteria
        sr_ = pd.Series([True] * len(df_), index=df_.index)
        if args.max_loss is not None:
            sr_ = sr_ & (df_['loss'] <= args.max_loss)
        if args.max_prob_dis is not None:
            sr_ = sr_ & (df_['prob_disc'] <= args.max_prob_dis)

        df_ = df_[sr_]

        # filter by top_n smallest
        df1_ = df_.nsmallest(20, 'prob_disc')
        df2_ = df_.nsmallest(20, 'loss')

        sequences = sequences | set(df1_['sequence']) | set(df2_['sequence'])

    # filter by repetition
    sequences = {sequence for sequence in sequences \
                 if not has_repeats(sequence, max_repeats=args.max_repeats)}

    # filter by cluster
    records = sequences2records(sequences)
    records_clust = easy_cluster(
        records,
        output_dir='.cluster',
        clean_up=True,
    )
    sequences_clust = records2sequences(records_clust)

    # filter by duplicate
    sr = (df['sequence'].isin(sequences_clust)) & \
        df['sequence'].duplicated()  # keep first duplicate occurrence

    # 3. (re)align to full database
    blast_dir.mkdir(exist_ok=True, parents=True)
    db_name = Path(args.target_db_path).name
    target_sequences = read_fasta(args.target_fasta, format='str')

    def _score(row):
        record = sequence2record(row['sequence'], str(row.get('id', None)))
        best_hit = blastp(
            records=[record],
            db_path=args.target_db_path,
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

        best_hit = {f'{db_name}_{key}': value for key, value in best_hit.items()}
        return pd.Series(best_hit)

    df.loc[sr, [
        f'{db_name}_pident',
        f'{db_name}_alignment_title',
        f'{db_name}_tseq',
        f'{db_name}_score',
        f'{db_name}_evalue',
    ]] = df[sr].progress_apply(_score, axis=1)

    # 4. prepare for alphafold run
    df_ = df[df['alphafold']]
    df_[f'{db_name}_pident_bins'] = pd.cut(
        df_[f'{db_name}_pident'],
        bins=pident_bins,
        labels=pident_labels,
        right=False,
    )

    df_['name'] = df_[f'{db_name}_pident_bins'] + '_' + pd.Series(list(df_.index))
    sequences_af = df_.set_index('name')['sequence'].to_dict()
    records = sequences2records(sequences_af)
    write_fasta(records, 'af.fasta')


if __name__ == '__main__':
    main()
