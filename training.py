import argparse
from pathlib import Path

from datasets import load_from_disk
import torch
from transformers import EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.models.esm import EsmForSequenceClassification, EsmConfig, EsmTokenizer
from transformers.utils import logging

from rubisgen.util import write_json
from rubisgen.configuration_progen import ProGenConfig
from rubisgen.data import DataCollatorWithPadding
from rubisgen.modeling_progen import ProGenForCausalLM
from rubisgen.tokenization_progen import create_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default=None, help='Output directory')
parser.add_argument('--dataset_dir', default=None, help='Dataset directory')
parser.add_argument('--fasta', default=None, help='Fasta file')
parser.add_argument('--lr', default=2e-4, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay')
parser.add_argument('--local_batch_size', default=2, type=int, help='Local batch size')
parser.add_argument('--global_batch_size', default=2, type=int, help='Global batch size')
parser.add_argument('--scratch_model_config', default=None, type=str, help='Scratch model config in json')
parser.add_argument('--max_steps', default=-1, type=int, help='Max steps; overrides --max_epochs')
parser.add_argument('--max_epochs', default=100, type=int, help='Max epochs')
parser.add_argument('--warmup_steps', default=None, type=int, help='(experimental) Warmup steps')
parser.add_argument('--logging_steps', default=10, type=int, help='Logging steps')
parser.add_argument('--save_steps', default=100, type=int, help='Save steps')
parser.add_argument('--debug', action='store_true', help='Debug mode')
parser.add_argument('--no_cuda', action='store_true', help='No CUDA mode')
parser.add_argument('--fp16', action='store_true', help='Use fp16')
parser.add_argument('--build_dataset_only', action='store_true', help='Terminate after building dataset')
parser.add_argument('--dataset_n_chunks', default=1, type=int, help='Build dataset in chunks')
parser.add_argument('--num_proc', default=None, type=int, help='num_proc for building dataset')
parser.add_argument('--test_ratio', default=0.1, type=float, help='Validation/Test ratio to training set')
parser.add_argument('--resume_from_checkpoint', default=None, type=str, help='Resume from training checkpoint')
parser.add_argument('--model_checkpoint', default=None, type=str, help='Finetune from checkpoint')
parser.add_argument('--is_discriminator', default=False, action='store_true', help='Train discriminator over generator')
args, unk = parser.parse_known_args()

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def main():
    # global config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(vars(args), output_dir / 'args.json')

    if unk:
        logger.warning(f'unknown args: {unk}')
    write_json(unk, output_dir / 'unk.json')

    dataset_dict = load_from_disk(args.dataset_dir)
    cols_keep = ['input_ids', 'attention_mask', 'labels']
    dataset_dict = dataset_dict.remove_columns([
        col for col in dataset_dict['train'].column_names if col not in cols_keep
    ])

    # training config
    if args.debug:
        local_batch_size = 1
        global_batch_size = torch.cuda.device_count() * local_batch_size if torch.cuda.is_available() \
            else local_batch_size
    else:
        local_batch_size = args.local_batch_size
        global_batch_size = args.global_batch_size

    grad_steps = global_batch_size // local_batch_size
    if torch.cuda.is_available():
        grad_steps = grad_steps // torch.cuda.device_count()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=args.logging_steps,
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        eval_accumulation_steps=1,
        save_total_limit=1,
        max_steps=1 if args.debug else args.max_steps,  # overrides num_train_steps
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=local_batch_size,
        per_device_eval_batch_size=local_batch_size,
        gradient_accumulation_steps=grad_steps,
        weight_decay=args.weight_decay,
        dataloader_num_workers=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        disable_tqdm=False,
        load_best_model_at_end=True,  # early stopping callback
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else 0,
        seed=42,
        learning_rate=args.lr,
        lr_scheduler_type='constant_with_warmup' if args.warmup_steps is not None else 'constant',
        no_cuda=args.no_cuda,
        fp16=args.fp16,
        prediction_loss_only=True,
        do_eval=True,
        ddp_find_unused_parameters=False,
        max_grad_norm=0.25,
    )

    # trainer config
    if args.is_discriminator:
        model_cls = EsmForSequenceClassification
        config_cls = EsmConfig
    else:
        model_cls = ProGenForCausalLM
        config_cls = ProGenConfig

    if args.model_checkpoint is None:
        logger.warning('model_checkpoint is not specified despite finetuning mode. Load from scratch instead.')
        if args.scratch_model_config:
            model_config = config_cls.from_json_file(args.scratch_model_config)
        else:
            model_config = config_cls()
        model = model_cls(model_config)
    else:
        model = model_cls.from_pretrained(args.model_checkpoint)

    if args.is_discriminator:
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t48_15B_UR50D')
    else:
        tokenizer = create_tokenizer()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['eval'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01)],
    )
    if args.resume_from_checkpoint is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    best_ckpt_dir = output_dir / 'checkpoint-best'
    trainer.model.save_pretrained(best_ckpt_dir)


if __name__ == '__main__':
    main()
