import argparse
import pandas as pd
from pathlib import Path
import os
from shutil import rmtree, copyfile

import numpy as np
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
    extract_bfactors,
)
from rubisgen.alignment.mmseqs import easy_cluster
from rubisgen.alignment.blast import blastp
from rubisgen.alignment.foldseek import easy_search

parser = argparse.ArgumentParser()
parser.add_argument('--input_dirs', required=True, nargs='+',
                    help='Directories containing input csv(s) / input csv(s)')
parser.add_argument('--output_dir', default='alphafold', help='Output directory')
parser.add_argument('--max_loss', type=float, default=None, help='Maximum loss')
parser.add_argument('--max_prob_disc', type=float, default=None, help='Maximum discriminator probability')
parser.add_argument('--pident_bins', type=float, default=[], nargs='+', help='A list of pident bins from mmseqs alignment')
parser.add_argument('--n_smallest', type=int, default=20, help='N smallest by loss or prob_disc')
parser.add_argument('--max_repeats', type=int, default=10, help='Maximum number of repeats')
parser.add_argument('--target_db_path', type=str, default=None, help='Path to blast database')
parser.add_argument('--target_fasta', type=str, default=None, help='Path to target fasta')
parser.add_argument('--num_threads', type=int, default=1, help='Number of threads')
parser.add_argument('--af_output_dir', default=None, help='Output directory from alphafold')
parser.add_argument('--db_name', default=None, help='Name of blast database')
parser.add_argument('--foldseek_db_path', default=None, help='Path to foldseek database')
args = parser.parse_args()

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def main():
    # 0. setup
    blast_dir = Path('.blast')
    if args.target_db_path is None:
        if args.db_name is None:
            raise ValueError('Either --target_db_path or --db_name must be specified.')
        db_name = args.db_name
    else:
        target_db_path = Path(args.target_db_path)
        db_name = target_db_path.name
        os.environ['BLASTDB'] = str(target_db_path.parent)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_fasta = output_dir / 'output.fasta'
    output_csv = output_dir / 'output.csv'
    
    if output_fasta.exists() and output_csv.exists():
        logger.info('Load in csv and fasta from previous run.')
        df = pd.read_csv(output_csv)
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
                df_af['csv'] = csv.name
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
            if args.pident_bins:
                sr_ = sr_ & (df_af['pident_bins'].isin(args.pident_bins)

            df_af = df_af[sr_]

            # filter by top_n smallest
            df1_ = df_af.nsmallest(args.n_smallest, 'prob_disc')
            df2_ = df_af.nsmallest(args.n_smallest, 'loss')

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
        if args.target_db_path is not None:
            blast_dir.mkdir(exist_ok=True, parents=True)

            if args.target_fasta is not None:
                logger.info(f'Loading in target fasta {args.target_fasta}.')
                target_sequences = read_fasta(args.target_fasta, format='str')

            def _score(row):
                record = sequence2record(row['sequence'], str(row.get('id', None)))
                best_hit = blastp(
                    records=[record],
                    db_path=args.target_db_path,
                    blastp_dir=blast_dir,
                    num_threads=args.num_threads,
                )

                alignment_hit_def = best_hit['alignment_hit_def']
                if alignment_hit_def is None or args.target_fasta is None:
                    best_hit['tseq'] = None
                else:
                    if not isinstance(alignment_hit_def, str):  # quickfix: type tuple
                        alignment_hit_def = alignment_hit_def[0]
                    best_hit['tseq'] = target_sequences[alignment_hit_def.split()[0]]

                best_hit = {f'{db_name}_{key}': value for key, value in best_hit.items()}
                return pd.Series(best_hit)

            if f'{db_name}_pident' in df.columns:  # skip calculated entries
                sr = sr & df[f'{db_name}_pident'].isna()

            logger.info(f'Begin alignment to {db_name}')
            tqdm.pandas(desc=f'Aligning to {db_name}')
            df.loc[sr, [
                f'{db_name}_pident',
                f'{db_name}_alignment_title',
                f'{db_name}_alignment_hit_def',
                f'{db_name}_tseq',
                f'{db_name}_score',
                f'{db_name}_evalue',
            ]] = df[sr].progress_apply(_score, axis=1)

        else:
            logger.warning('No blast db given. Skip alignment.')

            db_name = 'none'
            for col in ('pident', 'alignment_title', 'alignment_hit_def', 'tseq', 'score', 'evalue'):
                if col in df.columns:
                    df.loc[sr, f'{db_name}_{col}'] = df.loc[sr, col]

        sr_ = ~df[f'{db_name}_pident'].isna()
        df.loc[sr_, f'{db_name}_pident_bins'] = pd.cut(
            df.loc[sr_, f'{db_name}_pident'],
            bins=pident_bins,
            labels=pident_labels,
            right=False,
        )

        def _name(row):
            if row[f'{db_name}_pident_bins'] is None:
                return None
            return f'{row[f"{db_name}_pident_bins"]}_{row["csv"][:-4]}_{row["id"]}'

        df.loc[sr_, 'name'] = df.loc[sr_].apply(_name, axis=1)
        df.to_csv(output_csv, index=False)

        # 4. prepare for alphafold run
        logger.info('Prepare AlphaFold inputs.')

        sequences_af = df[sr_].set_index('name')['sequence'].to_dict()
        records_af = sequences2records(sequences_af)
        write_fasta(output_fasta, records_af.values())

        fasta_dir = output_dir / 'fasta'
        fasta_dir.mkdir(exist_ok=True)

        separate_fasta(
            fasta=output_fasta,
            dest_dir=fasta_dir,
            n_mer=2,
        )

    if 'plddt' in df.columns and 'pident_fs' in df.columns:
        # no need to re-calculate
        logger.info(f'plddt and pident_fs already calculated. Skip.')
        return

    if args.af_output_dir is None:
        logger.info('AlphaFold output not given. Ends here.')
        return

    # for backward compatibility
    if 'name' not in df.columns:
        sr_ = ~df[f'{db_name}_pident'].isna()
        pident_bins = [0, 30, 40, 50, 60, 70, 100]
        pident_labels = ['0-30', '30-40', '40-50', '50-60', '60-70', '70-100']

        df.loc[sr_, f'{db_name}_pident_bins'] = pd.cut(
            df.loc[sr_, f'{db_name}_pident'],
            bins=pident_bins,
            labels=pident_labels,
            right=False,
        )

        def _name(row):
            row = row.to_dict()
            bin_ = row[f'{db_name}_pident_bins']

            if bin_ is None or bin_ != bin_:
                return None
            return f'{row[f"{db_name}_pident_bins"]}_{row["csv"][:-4]}_{row["id"]}'

        df['name'] = df.apply(_name, axis=1)

    logger.info('Begin plddt calculation.')

    if args.af_output_dir is None:
        raise ValueError('Please specify --af_output_dir.')
    alphafold_dir = Path(args.af_output_dir)

    plddt = {}
    df_fs = {}
    df_ = None

    subdirs = sorted(subdir for subdir in alphafold_dir.glob('*') if subdir.is_dir())
    for subdir in tqdm(subdirs, desc='plddt and foldseek'):
        k = subdir.name.replace('output_', '')

        tmp_dir = Path('.foldseek')
        tmp_subdir = tmp_dir / 'pdb'
        tmp_dir.mkdir(exist_ok=True)
        tmp_subdir.mkdir(exist_ok=True)

        # plddt
        bfactors = []
        for pdb in subdir.glob('ranked_*.pdb'):
            copyfile(pdb, tmp_subdir / pdb.name)
            bfactor = np.mean(extract_bfactors(pdb))
            bfactors.append(bfactor)

        if not bfactors:
            # raise ValueError(f'No pdb found in {subdir}.')
            logger.warning(f'No pdb found in {subdir}. Skip.')
            continue

        plddt[k] = np.mean(bfactors)

        # foldseek
        foldseek_path = Path(args.foldseek_db_path)
        df_ = easy_search(
            query_dir=tmp_subdir,
            query_db=tmp_dir / 'query_db',
            pregenerated_target_db=foldseek_path,
            result=tmp_dir / 'result',
            ignore_empty=True,
        )
        df_.columns = [f'{col}_fs' for col in df_.columns]
        rmtree(tmp_dir)

        if df_['fident_fs'].isna().all() or df_.empty:
            df_fs[k] = {
                'query_fs': None,
                'target_fs': None,
                'fident_fs': 0,
                'alnlen_fs': None,
                'mismatch_fs': None,
                'gapopen_fs': None,
                'qstart_fs': None,
                'qend_fs': None,
                'tstart_fs': None,
                'tend_fs': None,
                'evalue_fs': None,
                'bits_fs': None,
            }
        else:
            idx = df_['fident_fs'].idxmax()
            df_fs[k] = df_.loc[idx].to_dict()

    if df_ is None:
        raise ValueError('No alphafold subdirectory found.')

    df['plddt'] = df['name'].apply(lambda x: plddt.get(x, None))
    for col in df_.columns:
        df[col] = df['name'].apply(lambda x: df_fs.get(x, {}).get(col, None))

    output_csv.rename(f'{output_csv}.bak')  # backup old csv
    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    main()
