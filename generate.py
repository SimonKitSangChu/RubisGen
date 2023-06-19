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
parser.add_argument('--top_p', default=0.75, type=float, help='Top p')
parser.add_argument('--min_length', default=300, type=int, help='Min length')
parser.add_argument('--max_length', default=500, type=int, help='Max length')
parser.add_argument('--num_return_sequences', default=100, type=int, help='Number of sequences to generate')
parser.add_argument('--generate_config', default=None, type=str, help='Generation config in json')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size in generation')
parser.add_argument('--start_tokens', default=None, type=str, help='Start tokens')
args = parser.parse_args()

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def main():
    # Load the model
    model = ProGenForCausalLM.from_pretrained(args.model_checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()

    # prepare generate config
    generate_config = {
        'do_sample': True,
        'max_length': args.max_length,
        'min_length': args.min_length,
        'top_p': args.top_p,
        'temperature': args.temperature,
    }
    if args.generate_config is not None:
        generate_config_ = read_json(args.generate_config)
        num_return_sequences = generate_config_.pop('num_return_sequences', args.num_return_sequences)
        if 'batch_size' in generate_config_:
            raise ValueError('batch_size should be parsed separately from generate_config')

        generate_config.update(generate_config_)

    # prepare start tokens
    start_tokens = generate_config.pop('start_tokens', args.start_tokens)
    if start_tokens is None:
        context = '1'
    else:
        context = '1' + start_tokens if start_tokens[0] != '1' else start_tokens

    tokenizer = create_tokenizer()
    input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(model.device)       

    # generate sequences
    sequences = []

    batch_size = min(args.batch_size, num_return_sequences)
    for _ in tqdm(range(num_return_sequences // batch_size), desc='Generating sequences'):
        output_ids = model.generate(
            input_ids,
            num_return_sequences=batch_size,
            **generate_config
        )
        output_ids = output_ids.cpu().numpy().tolist()
        sequences_ = tokenizer.decode_batch(output_ids)
        sequences_ = [s.lstrip('1').rstrip('2') for s in sequences_]  # drop BOS and EOS
        sequences_ = [s for s in sequences_ if '1' not in s and '2' not in s]  # BOS and EOS in main sequence
        sequences.extend(sequences_)

    # dump sequences
    records = sequences2records(sequences)
    write_fasta(args.output_fasta, records)


if __name__ == '__main__':
    main()
