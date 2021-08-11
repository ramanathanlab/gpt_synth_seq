from transformers import (
    # GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    # Trainer,
    # TrainingArguments,
    # DataCollatorForLanguageModeling,
    TextGenerationPipeline,
)

from Bio.Seq import Seq


def generate(model_dir, tokenizer_file, num_seqs=10):
    # loading things in to create pipeline
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        add_special_tokens=False,
        padding=True,
        max_length=1024,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # initialize the pipeline
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    seqs = []
    for i in range(num_seqs):
        s = pipeline("ATG")[0]["generated_text"]
        s = "".join(s.split())
        bio_seq = Seq(s)
        stop_locations = []
        for stop_codon in ["TAG", "TAA", "TGA"]:
            stop_index = bio_seq.find(stop_codon)
            if stop_index != -1:
                stop_locations.append(stop_index)
            # check if negative one

        try:
            min_stop_location = min(stop_locations)
        except ValueError:
            min_stop_location = None

        if min_stop_location:
            # save the stop codon
            bio_seq = bio_seq[: min_stop_location + 3]

        seqs.append(bio_seq)
