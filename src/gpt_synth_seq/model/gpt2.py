"""Generates a dataset, trains GPT-2 using huggingface from scratch"""
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# import transformers
# from tokenizers import Tokenizer
from gpt_synth_seq.data.dataset import get_dataset
import os
import argparse
from gpt_synth_seq.model.config import GPTConfig


def train(config):
    """Full training loop"""
    # get all fasta files in directory of training materials
    fasta_files = []
    for f in os.listdir(config.data_dir):
        if ".fasta" in f:
            fasta_files.append(str(os.path.join(config.data_dir, f)))

    dataset = get_dataset(
        fasta_files,
        config.processed_file_name,
        append=config.append_new_files,
        tmp_caching_location=config.tmp_caching_location,
        tokenizer_file=config.tokenizer_file_location,
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=config.tokenizer_file_location,
        add_special_tokens=False,
        padding=True,
        max_length=1024,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    config = GPT2Config(
        vocab_size=tokenizer.get_vocab_size(), n_positions=1024, max_length=1024
    )
    model = GPT2LMHeadModel(config)

    training_args = TrainingArguments(
        output_dir=config.output_model_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        save_steps=10_000,
        prediction_loss_only=True,
        save_total_limit=2,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("Finished training.")
    trainer.save_model()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = GPTConfig.from_yaml(args.config)
    train(cfg)
