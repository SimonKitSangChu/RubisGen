import argparse
import pandas as pd
from pathlib import Path
import os

from transformers.utils import logging
from tqdm import tqdm

from rubisgen.util import (
    read_fasta,
    write_fasta,
    separate_fasta,
    sequence2record,
    sequences2records,
    records2sequences,
    has_repeats,
)
from rubisgen.alignment.mmseqs import easy_cluster
from rubisgen.alignment.blast import blastp

parser = argparse.ArgumentParser()
parser.add_argument('--input_dirs', required=True, nargs='+',
                    help='Directories containing input csv(s) / input csv(s)')
parser.add_argument('--output_dir', default='alphafold', help='Output directory')
parser.add_argument('--max_loss', type=float, default=None, help='Maximum loss')
parser.add_argument('--max_prob_disc', type=float, default=None, help='Maximum discriminator probability')
parser.add_argument('--max_repeats', type=int, default=10, help='Maximum number of repeats')
parser.add_argument('--target_db_path', type=str, default=None, help='Path to blast database')
parser.add_argument('--target_fasta', type=str, default=None, help='Path to target fasta')
parser.add_argument('--num_threads', type=int, default=1, help='Number of threads')
args = parser.parse_args()

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

blast_dir = Path('.blast')
os.environ['BLASTDB'] = str(blast_dir)


def main():
    # 0. prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_fasta = output_dir / 'output.fasta'
    output_csv = output_dir / 'output.csv'
    
    if output_fasta.exists() and output_csv.exists():
        df = pd.read_csv(output_csv)
        records_af = read_fasta(output_fasta)
    else:
        # 1. load input csv(s)
        df = []
        for input_dir in args.input_dirs:
            input_dir = Path(input_dir)

            if input_dir.is_file() and input_dir.suffix == '.csv':
                csv_list = [input_dir]
            else:
                csv_list = input_dir.glob('*.csv')

            for csv in csv_list:
                df_af = pd.read_csv(csv)
                assert 'sequence' in df_af.columns, f'No column sequence in {csv}'
                df.append(df_af)

        df = pd.concat(df)

        # 2. filtering
        pident_bins = [0, 30, 40, 50, 60, 70, 100]
        pident_labels = ['0-30', '30-40', '40-50', '50-60', '60-70', '70-100']
        df['pident_bins'] = pd.cut(df['pident'], bins=pident_bins, labels=pident_labels, right=False)

        sequences = set()
        for pident_bin, df_af in df.groupby('pident_bins'):
            # filter by criteria
            sr_ = pd.Series([True] * len(df_af), index=df_af.index)
            if args.max_loss is not None:
                sr_ = sr_ & (df_af['loss'] <= args.max_loss)
            if args.max_prob_disc is not None:
                sr_ = sr_ & (df_af['prob_disc'] <= args.max_prob_disc)

            df_af = df_af[sr_]

            # filter by top_n smallest
            df1_ = df_af.nsmallest(20, 'prob_disc')
            df2_ = df_af.nsmallest(20, 'loss')

            sequences = sequences | set(df1_['sequence']) | set(df2_['sequence'])

        # filter by repetition
        sequences = {sequence for sequence in sequences \
                     if not has_repeats(sequence, max_repeats=args.max_repeats)}

        # filter by cluster
        records_clust = easy_cluster(
            sequences,
            output_dir='.cluster',
            clean_up=True,
        )
        sequences_clust = records2sequences(records_clust)
        sequences_clust = set(sequences_clust.values())

        # filter by duplicate
        sr = (df['sequence'].isin(sequences_clust)) & \
            ~df['sequence'].duplicated()  # keep first duplicate occurrence

        # 3. (re)align to full database
        blast_dir.mkdir(exist_ok=True, parents=True)
        target_db_path = Path(args.target_db_path)
        db_name = target_db_path.name

        if args.target_fasta is None:
            target_fasta = target_db_path.with_stem(f'{db_name}.fasta')
        else:
            target_fasta = Path(args.target_fasta)
        target_sequences = read_fasta(target_fasta, format='str')

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

        tqdm.pandas(desc=f'Aligning to {db_name}')
        df.loc[sr, [
            f'{db_name}_pident',
            f'{db_name}_alignment_title',
            f'{db_name}_tseq',
            f'{db_name}_score',
            f'{db_name}_evalue',
        ]] = df[sr].progress_apply(_score, axis=1)
        df.to_csv(output_csv, index=False)

        # 4. prepare for alphafold run
        df_af = df.dropna(subset=[f'{db_name}_tseq']).copy()
        df_af[f'{db_name}_pident_bins'] = pd.cut(
            df_af[f'{db_name}_pident'],
            bins=pident_bins,
            labels=pident_labels,
            right=False,
        )

        df_af['name'] = df_af[f'{db_name}_pident_bins'].astype(str) + '_' + df_af['id'].astype(str)
        sequences_af = df_af.set_index('name')['sequence'].to_dict()
        records_af = sequences2records(sequences_af)
        write_fasta(output_fasta, records_af.values())

        fasta_dir = output_dir / 'fasta'
        fasta_dir.mkdir(exist_ok=True)

        separate_fasta(
            fasta=output_fasta,
            dest_dir=fasta_dir,
            n_mer=2,
        )


if __name__ == '__main__':
    main()
