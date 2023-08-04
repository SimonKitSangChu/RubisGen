from os import PathLike
from pathlib import Path
from io import StringIO
from typing import Any, Dict

from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbimakeblastdbCommandline, NcbiblastpCommandline
from Bio.SeqRecord import SeqRecord

from rubisgen.util import write_fasta


def create_blastdb(fasta_file: PathLike, db_name: PathLike):
    db_name = Path(db_name)
    db_dir = db_name.parent
    if any('psq' in filename for filename in db_dir.glob(db_name.name + '.*')):
        return

    cmd = NcbimakeblastdbCommandline(dbtype='prot', input_file=fasta_file, out=db_name)
    cmd()


def blastp(
        records: Dict[str, SeqRecord],
        db_path: PathLike,
        blastp_dir: PathLike = '.blastp',
        max_e_value: float = 1e-6,
        **kwargs,
) -> Dict[str, Any]:
    blastp_dir = Path(blastp_dir)
    blastp_dir.mkdir(exist_ok=True, parents=True)

    fasta = blastp_dir / 'query.fasta'
    write_fasta(fasta, records)

    db_path = Path(db_path)
    blast_cmd = NcbiblastpCommandline(
        query=fasta,
        db=db_path.name,
        outfmt=5,
        **kwargs,
    )
    output = blast_cmd()[0]

    best_hit = {
        'pident': 0,
        'alignment_title': None,
        'alignment_hit_def': None,
        'tseq': None,
        # 'tseq_gapped': None,
        'score': None,
        'evalue': None,
    }
    blast_records = NCBIXML.read(StringIO(output))

    for alignment in blast_records.alignments:
        for hsp in alignment.hsps:
            alignment_length = hsp.align_length
            identities = hsp.identities
            percent_identity = (identities / alignment_length) * 100

            if percent_identity > best_hit['pident'] and hsp.expect < max_e_value:
                best_hit['pident'] = percent_identity
                best_hit['alignment_title'] = alignment.title
                best_hit['alignment_hit_def'] = alignment.hit_def.replace('<unknown description>', '').strip()
                # best_hit['tseq_gapped'] = str(hsp.sbjct)
                best_hit['score'] = hsp.score
                best_hit['evalue'] = hsp.expect

    return best_hit

