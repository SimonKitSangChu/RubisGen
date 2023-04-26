import argparse
import torch
from tqdm import tqdm

from transformers.utils import logging

from rubisgen.tokenization_progen import create_tokenizer
from rubisgen.modeling_progen import ProGenForCausalLM
from rubisgen.util import write_fasta, sequences2records, read_json

parser = argparse.ArgumentParser()
parser.add_argument('--output_fasta', default=None, help='Output fasta file')
parser.add_argument('--model_checkpoint', default=None, type=str, help='Model checkpoint')
parser.add_argument('--temperature', default=1., type=float, help='Temperature')
parser.add_argument('--top_p', default=0.9, type=float, help='Top p')
parser.add_argument('--min_length', default=300, type=int, help='Min length')
parser.add_argument('--max_length', default=500, type=int, help='Max length')
parser.add_argument('--num_return_sequences', default=100, type=int, help='Number of sequences to generate')
parser.add_argument('--generation_config', default=None, type=str, help='Generation config in json')
parser.add_argument('--start_tokens', default=None, type=str, help='Start tokens')
args, unk = parser.parse_known_args()

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def main():
    # Load the model
    model = ProGenForCausalLM.from_pretrained(args.model_checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()

    if args.start_tokens is None:
        context = '1'
    else:
        context = '1' + args.start_tokens if args.start_tokens[0] != '1' else args.start_tokens

    tokenizer = create_tokenizer()
    input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(model.device)

    # Generate sequences
    generation_config = {
        'do_sample': True,
        'max_length': args.max_length,
        'min_length': args.min_length,
        'top_p': args.top_p,
    }
    if args.generation_config is not None:
        generation_config.update(
            read_json(args.generation_config)
        )

    sequences = []
    for _ in tqdm(range(args.num_return_sequences // args.batch_size), desc='Generating sequences'):
        output_ids = model.generate(
            input_ids,
            temperature=args.temperature,
            num_return_sequences=args.batch_size,
            **generation_config
        )
        output_ids = output_ids.cpu().numpy().tolist()
        sequences_ = tokenizer.decode_batch(output_ids)
        sequences_ = [s.lstrip('1').rstrip('2') for s in sequences_]
        sequences.extend(sequences_)

    records = sequences2records(sequences)
    write_fasta(args.output_fasta, records)


if __name__ == '__main__':
    main()
