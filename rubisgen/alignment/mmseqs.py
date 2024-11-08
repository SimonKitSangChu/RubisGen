from os import PathLike
from pathlib import Path
from shutil import rmtree
import subprocess as sp
from typing import Any, Dict, Iterable, Optional

from Bio.SeqRecord import SeqRecord
import pandas as pd
from transformers.utils import logging

from rubisgen.util import write_fasta, sequences2records, read_fasta


def kwargs2flags(kwargs: Dict[str, Any]) -> str:
    if not kwargs:
        return ''
    return ' '.join([f'--{k} {v}' for k, v in kwargs.items()])


def create_db(fasta: PathLike, db: PathLike, **kwargs) -> None:
    cmd = f'mmseqs createdb {fasta} {db}' + kwargs2flags(kwargs)
    sp.run(cmd, shell=True, check=True)


def search(queryDB: PathLike, targetDB: PathLike, resultDB: PathLike,
           tmp_dir: PathLike = '.tmp', clean: bool = True, **kwargs) -> None:
    cmd = f'mmseqs search {queryDB} {targetDB} {resultDB} {tmp_dir}' + kwargs2flags(kwargs)
    sp.run(cmd, shell=True, check=True)
    if clean:
        rmtree(tmp_dir)


def alignall(queryDB: PathLike, targetDB: PathLike, resultDB: PathLike = 'alignall', **kwargs) -> None:
    # (How to create a fake prefiltering for all-vs-all alignments)
    # reference: https://github.com/soedinglab/mmseqs2/wiki
    queryDB = Path(queryDB).resolve()
    targetDB = Path(targetDB).resolve()
    resultDB = Path(resultDB).resolve()

    # check if all DBs are in the same directory
    assert queryDB.parent == targetDB.parent == resultDB.parent, 'All DB must be in the same directory!'
    dirname = queryDB.parent
    queryDB, targetDB, resultDB = queryDB.name, targetDB.name, resultDB.name

    cmd = f'''
    cd {dirname};
    ln -s {targetDB}.index {resultDB}_pref;
    INDEX_SIZE="$(echo $(wc -c < "{targetDB}.index"))";
    awk -v size=$INDEX_SIZE '{{ print $1"\t0\t"size; }}' "{queryDB}.index" > "{resultDB}_pref.index";
    awk 'BEGIN {{ printf("%c%c%c%c",7,0,0,0); exit; }}' > "{resultDB}_pref.dbtype";
    mmseqs align "{queryDB}" "{targetDB}" "{resultDB}_pref" "{resultDB}" {kwargs2flags(kwargs)};
    '''
    sp.run(cmd, shell=True, check=True)


def convertalis(queryDB: PathLike, targetDB: PathLike, resultDB: PathLike, result: PathLike, **kwargs) -> None:
    kwargs['format-output'] = kwargs.get(
        'format-output', 'query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qseq,tseq'
    )
    cmd = f'mmseqs convertalis {queryDB} {targetDB} {resultDB} {result}.m8 ' + kwargs2flags(kwargs)
    sp.run(cmd, shell=True, check=True)


def parse_m8(m8: PathLike, format_columns: Optional[Iterable] = None) -> pd.DataFrame:
    if format_columns is None:
        format_columns = [
            'query', 'target', 'pident', 'alnlen', 'mismatch', 'gapopen', 'qstart', 'qend', 'tstart', 'tend', 'evalue',
            'bits', 'qseq', 'tseq'
        ]

    try:
        df = pd.read_csv(m8, sep='\t', header=None)
        df.columns = format_columns
        return df
    except pd.errors.EmptyDataError:
        logger.warning(f'{m8} is empty')
        return pd.DataFrame()


def pairwise_align(
        query_records: Iterable[SeqRecord],
        target_records: Optional[Iterable[SeqRecord]] = None,
        query_db: str = 'queryDB',
        target_db: str = 'targetDB',
        result_db: str = 'resultDB',
        result: str = 'result',
        all_to_all: bool = False,
        drop_self: bool = False,
) -> pd.DataFrame:
    write_fasta(query_db, query_records)
    create_db(query_db, query_db)

    if target_records is None:
        target_db = query_db
    else:
        write_fasta(target_db, target_records)
        create_db(target_db, target_db)

    if all_to_all:
        alignall(query_db, target_db, result_db)
    else:
        search(query_db, target_db, result_db)

    convertalis(query_db, target_db, result_db, result)
    df = parse_m8(str(result) + '.m8')

    if drop_self:
        df = df[df['query'] != df['target']]

    return df


def easy_cluster(
        records: Iterable[SeqRecord],
        output_dir: PathLike = 'mmseqs_cluster',
        clean_up: bool = True,
        **kwargs,
) -> Iterable[SeqRecord]:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    records = sequences2records(records)
    write_fasta(output_dir / 'mmseqs_cluster.fasta', records)

    kwargs['min-seq-id'] = kwargs.pop('min_seq_id', 0.8)
    cmd = f'mmseqs easy-cluster {kwargs2flags(kwargs)} {output_dir}/mmseqs_cluster.fasta {output_dir}/clustered {output_dir}/tmp'
    try:
        sp.run(cmd, shell=True, check=True, stdout=sp.PIPE, stderr=sp.PIPE)
    except sp.CalledProcessError as e:
        raise RuntimeError(f'Error in mmseqs easy-cluster: {e}')

    records = read_fasta(output_dir / 'clustered_rep_seq.fasta')
    if clean_up:
        rmtree(output_dir)

    return records
