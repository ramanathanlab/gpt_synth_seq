"""Utilities for creation of dataset to be used with Huggingface training methods from FASTA files"""
from Bio import SeqIO
from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_into_codons(sequence):
    """Given a sequence, split it into it's codons"""
    split_string = ""
    for x in chunks(sequence, 3):
        split_string += x
        split_string += " "
    return split_string


def whitespace_codon_split_generator(records):
    """Given a list of SeqIO records, yield strings which are split into codons by whitespace, must be divisible by 3"""
    for s in tqdm(records):
        if len(str(s.seq)) % 3 == 0:
            yield split_into_codons(str(s.seq))
        else:
            print("Skipping a sequence which is not divisble by 3...")
            pass


def fasta_to_codon_splits(filename):
    """Given a fasta file, return the generator of string sequences"""
    print("Generating codon splits for fasta file at {}".format(filename))
    records = list(SeqIO.parse(filename, "fasta"))
    return whitespace_codon_split_generator(records)

def generate_processed_text_file(fasta_files:list, processed_filename:str, append: bool=False):
    if append:
        print("Note: appending to existing processed sequences files. Set append=False to overwrite.")
        f = open(processed_filename, 'a')
    else:
        f = open(processed_filename, 'w')
    for fasta in tqdm(fasta_files, desc="Fasta File Processing"):
        processed_sequences = fasta_to_codon_splits(fasta)
        for i in processed_sequences:
            f.write(i)
            f.write("\n")
    f.close()
    return