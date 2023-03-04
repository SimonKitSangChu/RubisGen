import torch
from tqdm import tqdm

from rubisgen.tokenization_progen import create_tokenizer
from rubisgen.modeling_progen import ProGenForCausalLM
from rubisgen.util import write_fasta, sequences2records

# Load the model
model = ProGenForCausalLM.from_pretrained('training/progen2-small/checkpoint-best')
if torch.cuda.is_available():
    model = model.cuda()

tokenizer = create_tokenizer()
context = '1'
input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(model.device)

# Generate sequences
generation_config = {
    'do_sample': True,
    'max_length': 600,
    'min_length': 100,
    'top_p': 0.9,
}

temperatures = (1., 1.5, 2.)
num_return_sequences = 100
batch_size = 20

for temperature in temperatures:
    sequences = []
    for _ in tqdm(range(num_return_sequences // batch_size), desc=f'Temperature = {temperature}'):
        output_ids = model.generate(
            input_ids,
            temperature=temperature,
            num_return_sequences=batch_size,
            **generation_config
        )
        output_ids = output_ids.cpu().numpy().tolist()
        sequences_ = tokenizer.decode_batch(output_ids)
        sequences_ = [s.lstrip('1').strip('2') for s in sequences_]
        sequences.extend(sequences_)

    records = sequences2records(sequences)
    write_fasta(f'temp_{temperature}.fasta', records)
