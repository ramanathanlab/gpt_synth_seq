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
from Bio import SeqIO
from argparse import ArgumentParser
from Bio.SeqRecord import SeqRecord
from transformers import set_seed


def generate(model_dir, tokenizer_file, output_file, protein_family, num_seqs=10):
    # loading things in to create pipeline
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        add_special_tokens=False,
        padding=True,
        max_length=1024,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    set_seed(0)
    # initialize the pipeline
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    # generate all sequences
    seqs = []
    print("Generating sequences...")
    generated = pipeline("ATG", num_return_sequences=num_seqs)
    for i in generated:
        s = i["generated_text"]
        s = "".join(s.split())
        bio_seq = Seq(s)
        stop_locations = []
        for stop_codon in ["TAG", "TAA", "TGA"]:
            stop_index = bio_seq.find(stop_codon)
            if stop_index != -1:
                stop_locations.append(stop_index)

        try:
            min_stop_location = min(stop_locations)
        except ValueError:
            min_stop_location = None

        if min_stop_location:
            # save the stop codon
            bio_seq = bio_seq[: min_stop_location + 3]

        record = SeqRecord(
            bio_seq,
            id="GENE{}".format(i),
            name=protein_family,
            description="Synthetic generated sequence for family {}".format(
                protein_family
            ),
        )

        seqs.append(record)

    # print sequences to file
    with open(output_file, "w") as output_handle:
        SeqIO.write(seqs, output_handle, "fasta")


if __name__ == "__main__":
    parser = ArgumentParser("Generate sequences using an existing trained model.")
    parser.add_argument("-m", "--model_dir", help="Directory for trained model dir")
    parser.add_argument(
        "-o",
        "--output_file",
        help="Filename to put generated sequences, must be fasta extension",
    )
    parser.add_argument("-t", "--tokenizer_file", default="codon_tokenizer.json")
    parser.add_argument("-p", "--protein_family")
    parser.add_argument("-n", "--num_seqs", default=10)
    args = parser.parse_args()
    generate(
        args.model_dir,
        args.tokenizer_file,
        args.output_file,
        args.protein_family,
        int(args.num_seqs),
    )
