import json
import yaml
from pydantic import BaseSettings as _BaseSettings
from pathlib import Path
from typing import TypeVar, Type, Union

_T = TypeVar("_T")


class BaseSettings(_BaseSettings):
    def dump_yaml(self, cfg_path):
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: Union[str, Path]) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class GPTConfig(BaseSettings):
    data_dir: Path = "/lambda_stor/projects/MDH-design/full_patric_download/"
    processed_file_name: Path = "/tmp/mzvyagin/patric_processed_sequences.txt"
    append_new_files: bool = False
    tmp_caching_location: Path = "/tmp/mzvyagin/gpt_dataset.pkl"
    tokenizer_file_location: Path = "codon_tokenizer.json"
    output_model_dir: Path = "gpt_model"
    epochs: int = 25
    batch_size: int = 32


if __name__ == "__main__":
    GPTConfig().dump_yaml("gpt_template.yaml")
