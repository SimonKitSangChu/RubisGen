from os import PathLike
from pathlib import Path
import subprocess as sp
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from pandas.errors import EmptyDataError
from transformers.utils import logging


def kwargs2flags(kwargs: Dict[str, Any]) -> str:
    if not kwargs:
        return ''
    return ' '.join([f'--{k} {v}' for k, v in kwargs.items()])


def create_db(input: PathLike, db: PathLike, **kwargs) -> None:
    cmd = f'foldseek createdb {input} {db}' + kwargs2flags(kwargs)
    sp.run(cmd, shell=True, check=True, stdout=sp.PIPE, stderr=sp.PIPE)


def search(queryDB: PathLike, targetDB: PathLike, result: PathLike, clean_tmp: bool = True, **kwargs) -> None:
    cmd = f'foldseek easy-search {queryDB} {targetDB} {result}.m8 tmp' + kwargs2flags(kwargs)
    sp.run(cmd, shell=True, check=True, stdout=sp.PIPE, stderr=sp.PIPE)
    if clean_tmp:
        sp.run('rm -rf tmp', shell=True, check=True)


def parse_m8(m8: PathLike, format_columns: Optional[Iterable] = None, ignore_empty: bool = False) -> pd.DataFrame:
    if format_columns is None:
        format_columns = 'query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits'.split(',')

    try:
        df = pd.read_csv(m8, sep='\t', header=None)
    except EmptyDataError:
        logger.warning(f'Emptpy foldseek result encountered. Skip.')
        df = pd.DataFrame([[None] * len(format_columns)])

    df.columns = format_columns
    return df


def easy_search(
        query_dir: PathLike,
        target_dir: Optional[PathLike] = None,
        query_db: str = 'queryDB',
        target_db: str = 'targetDB',
        pregenerated_target_db: Optional[PathLike] = None,
        result: str = 'result',
        drop_self: bool = True,
        ignore_empty: bool = False,
) -> pd.DataFrame:
    create_db(query_dir, query_db)

    if pregenerated_target_db is not None:
        target_db = pregenerated_target_db
    elif target_dir is None:
        target_db = query_db
    else:
        create_db(target_dir, target_db)

    search(query_dir, target_db, result)
    df = parse_m8(str(result) + '.m8', ignore_empty=ignore_empty)

    if drop_self:
        df = df[df['query'] != df['target']]

    return df
